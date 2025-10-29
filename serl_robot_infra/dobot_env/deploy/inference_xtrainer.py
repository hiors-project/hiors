#!/usr/bin/env python3
import time
import math
import threading
import argparse
import numpy as np
import rospy
import cv2
from scipy.interpolate import interp1d
from data_msgs.msg import Event
from self_defined_classes import RobotAction, ComponentAction, RobotRosClient
from openpi_client import websocket_client_policy

url: str = "your_url"
policy_client = websocket_client_policy.WebsocketClientPolicy(url)
_G_ACTION_LIST = []
_G_STATE_LIST = []
_G_LAST_EVENT_DETAIL = None
_G_EVENT_LOCK = threading.Lock()


def event_callback(msg):
    global _G_LAST_EVENT_DETAIL
    with _G_EVENT_LOCK:
        _G_LAST_EVENT_DETAIL = msg.event_detail


def get_server_teleoperation_flag():
    with _G_EVENT_LOCK:
        return _G_LAST_EVENT_DETAIL


def obs_to_states(obs, no_image: bool = False):
    if not no_image:
        cam_left_wrist = np.transpose(cv2.resize(obs.chain_images[1].color_image, (298, 224)), (2, 0, 1))
        cam_right_wrist = np.transpose(cv2.resize(obs.chain_images[2].color_image, (298, 224)), (2, 0, 1))
        resized_color = cv2.resize(obs.chain_images[0].color_image, (298, 224))
        cam_high = np.transpose(resized_color, (2, 0, 1))
    else:
        cam_left_wrist = None
        cam_right_wrist = None
        cam_high = None

    obs_ee_pose_left = np.array(obs.get_observation(name='left_arm').multibody_state)  # (7,)
    obs_gripper_left = obs.get_observation(name='left_hand').multibody_state  # (1,)
    obs_ee_pose_right = np.array(obs.get_observation(name='right_arm').multibody_state)  # (7,)
    obs_gripper_right = obs.get_observation(name='right_hand').multibody_state  # (1,)

    state = np.concatenate((obs_ee_pose_left, obs_gripper_left, obs_ee_pose_right, obs_gripper_right), axis=0)
    server_teleoperation_flag = get_server_teleoperation_flag()
    return state, cam_high, cam_left_wrist, cam_right_wrist, server_teleoperation_flag


def infer(obs, policy_client, use_tactile=False):  # use_tactile kept for interface consistency
    state, cam_high, cam_left_wrist, cam_right_wrist, server_teleoperation_flag = obs_to_states(obs)

    lock.acquire()
    state_chunk = _G_STATE_LIST.copy()
    _G_STATE_LIST.clear()
    lock.release()

    element = {
        "state": state,
        "images": {
            "cam_high": cam_high,
            "cam_left_wrist": cam_left_wrist,
            "cam_right_wrist": cam_right_wrist,
        },
        "prompt": "put the objects in the box",
        "server_teleoperation_flag": server_teleoperation_flag,
        "state_chunk": state_chunk,
    }

    t0 = time.time()
    action_chunk = policy_client.infer(element)["actions"]
    print('action_chunk', action_chunk.shape, 'time', time.time() - t0)
    return action_chunk, state, server_teleoperation_flag


def generate_action(cur_action):
    actions = RobotAction()
    act_ee_pose_left = cur_action[0:6]
    act_gripper_left = cur_action[6]
    act_ee_pose_right = cur_action[7:13]
    act_gripper_right = cur_action[13]

    left_arm_action = ComponentAction()
    left_arm_action.name = 'left_arm'
    left_arm_action.joint_commands = np.array(act_ee_pose_left)
    left_arm_action.pose_command = None
    left_arm_action.duration = 0.0
    actions.actions.append(left_arm_action)

    right_arm_action = ComponentAction()
    right_arm_action.name = 'right_arm'
    right_arm_action.joint_commands = np.array(act_ee_pose_right)
    right_arm_action.pose_command = None
    right_arm_action.duration = 0.0
    actions.actions.append(right_arm_action)

    left_hand_action = ComponentAction()
    left_hand_action.name = 'left_hand'
    left_hand_action.joint_commands = np.array([act_gripper_left])
    left_hand_action.pose_command = None
    left_hand_action.duration = 0.0
    actions.actions.append(left_hand_action)

    right_hand_action = ComponentAction()
    right_hand_action.name = 'right_hand'
    right_hand_action.joint_commands = np.array([act_gripper_right])
    right_hand_action.pose_command = None
    right_hand_action.duration = 0.0
    actions.actions.append(right_hand_action)

    actions.timestamp = rospy.Time.now().to_sec()
    return actions


def interpolate_joints(joints, factor=5):
    """Linear interpolation over joint sequence.
    joints: (N, D); returns (N*factor, D)."""
    N, _ = joints.shape
    x_old = np.arange(N)
    x_new = np.linspace(0, N - 1, N * factor)
    f = interp1d(x_old, joints, axis=0, kind='linear')
    return f(x_new)


def deploy_policy(client, rate, use_tactile=False):
    ros_rate = rospy.Rate(rate)
    while not rospy.is_shutdown():
        obs = client.get_observation(display_image=False)
        action_chunk_before, state, server_teleoperation_flag = infer(obs, policy_client, use_tactile)
        if server_teleoperation_flag is None or server_teleoperation_flag == 'teleoperation to server':
            chunk_size = 20
        elif server_teleoperation_flag == 'server to teleoperation':
            chunk_size = 1
        else:
            chunk_size = 20  # fallback
        action_chunk_before = np.concatenate([state[None, :], action_chunk_before], axis=0)
        action_chunk = interpolate_joints(action_chunk_before, factor=5)
        for act_i in range(chunk_size * 5):
            actions = generate_action(action_chunk[act_i])
            client.send_action(actions)
            ros_rate.sleep()
    print('deploy policy done.')


def policy_infer_thread(client, infer_rate, act_rate, use_tactile=False):
    cur_obs_t = time.time()
    global _G_ACTION_LIST, _G_STATE_LIST
    while not rospy.is_shutdown():
        loop_start = time.time()
        obs = client.get_observation(display_image=False)
        print(f'obs: {time.time() - cur_obs_t}')
        cur_obs_t = time.time()
        infer_start = time.time()
        action_chunk = infer(obs, policy_client, use_tactile)[0].tolist()
        print(f'infer: {time.time() - infer_start}')
        lock.acquire()
        if len(_G_ACTION_LIST) == 0:
            _G_ACTION_LIST = action_chunk.copy()
        else:
            removed_len = math.ceil((time.time() - cur_obs_t) * act_rate)
            if removed_len < len(action_chunk):
                _G_ACTION_LIST = action_chunk[removed_len:]
            else:
                print(f'Predicted action length {len(action_chunk)} too small')
        lock.release()
        sleep_sec = (1 / infer_rate) - (time.time() - loop_start)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
            print(f'sleep: {sleep_sec}')
        print(f'loop: {time.time() - loop_start}')
    print('deploy policy done.')


def send_action_thread(client, rate):
    global _G_ACTION_LIST
    ros_rate = rospy.Rate(rate)
    while not rospy.is_shutdown():
        ros_rate.sleep()
        lock.acquire()
        cur_action = _G_ACTION_LIST.pop(0) if _G_ACTION_LIST else None
        lock.release()
        if cur_action is None:
            print('Empty action list')
            continue
        if isinstance(cur_action, RobotAction):
            actions = cur_action
            actions.timestamp = rospy.Time.now().to_sec()
        else:
            actions = generate_action(cur_action)
        client.send_action(actions)
    print('send action done.')


def store_state_thread(client, rate):
    global _G_STATE_LIST
    ros_rate = rospy.Rate(rate)
    while not rospy.is_shutdown():
        obs = client.get_observation(display_image=False)
        state, _, _, _, server_teleoperation_flag = obs_to_states(obs, no_image=True)
        state_timestamp = rospy.Time.now().to_sec()
        state_dict = {
            'state': state,
            'server_teleoperation_flag': server_teleoperation_flag,
            'state_timestamp': state_timestamp,
        }
        ros_rate.sleep()
        lock.acquire()
        _G_STATE_LIST.append(state_dict)
        lock.release()


if __name__ == '__main__':
    rospy.init_node('robot_ros_client_example', anonymous=True)
    rospy.loginfo("ROS node initialized")

    rospy.Subscriber('/robot/data/xtrainer/event', Event, event_callback)

    parser = argparse.ArgumentParser(description='Realtime policy deployment (no rosbag).')
    parser.add_argument('--rate', type=float, help='action execution rate', default=50)
    parser.add_argument('--parallel', type=bool, help='run inference and execution in parallel', default=False)
    _ARGS, _ = parser.parse_known_args()

    client = RobotRosClient(frame_id='robot_ros_client')
    lock = threading.Lock()

    if _ARGS.parallel:
        print('parallel inference')
        action_thread = threading.Thread(target=send_action_thread, args=(client, _ARGS.rate))
        action_thread.daemon = True
        action_thread.start()

        policy_thread = threading.Thread(target=policy_infer_thread, args=(client, 4.0, _ARGS.rate))
        policy_thread.daemon = True
        policy_thread.start()

        store_thread = threading.Thread(target=store_state_thread, args=(client, 10))
        store_thread.daemon = True
        store_thread.start()

        rospy.spin()
        action_thread.join()
        policy_thread.join()
        store_thread.join()
    else:
        print('serial inference')
        store_thread = threading.Thread(target=store_state_thread, args=(client, 10))
        store_thread.daemon = True
        store_thread.start()

        policy_thread = threading.Thread(target=deploy_policy, args=(client, _ARGS.rate, False))
        policy_thread.daemon = True
        policy_thread.start()

        store_thread.join()
        policy_thread.join()
