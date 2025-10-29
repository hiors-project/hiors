import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from functools import partial
import math


def random_crop(img, *, padding):
    """
    Randomly crop an image with padding.
    
    Args:
        img: Image tensor of shape [..., H, W, C]
        padding: Padding amount
        
    Returns:
        Cropped image of the same shape as input
    """
    # Get original shape
    original_shape = img.shape
    h, w = original_shape[-3], original_shape[-2]
    
    # Generate random crop coordinates
    crop_from_h = torch.randint(0, 2 * padding + 1, (1,)).item()
    crop_from_w = torch.randint(0, 2 * padding + 1, (1,)).item()
    
    # Pad the image
    # PyTorch pad format is (left, right, top, bottom, front, back, ...)
    padded_img = F.pad(img, (0, 0, padding, padding, padding, padding), mode='replicate')
    
    # Crop from the padded image
    cropped = padded_img[..., crop_from_h:crop_from_h + h, crop_from_w:crop_from_w + w, :]
    
    return cropped


def batched_random_crop(img, padding, num_batch_dims: int = 1):
    """
    Apply random crop to a batch of images.
    
    Args:
        img: Image tensor of shape [B, T, H, W, C] or similar
        padding: Padding amount
        num_batch_dims: Number of batch dimensions
        
    Returns:
        Batch of cropped images with same shape as input
    """
    # Flatten batch dims
    original_shape = img.shape
    batch_size = 1
    for i in range(num_batch_dims):
        batch_size *= img.shape[i]
    
    img_flat = img.reshape(batch_size, *img.shape[num_batch_dims:])
    
    # Apply random crop to each image in the batch
    cropped_list = []
    for i in range(batch_size):
        cropped = random_crop(img_flat[i], padding=padding)
        cropped_list.append(cropped)
    
    # Stack and reshape back to original batch dimensions
    cropped_batch = torch.stack(cropped_list, dim=0)
    cropped_batch = cropped_batch.reshape(original_shape)
    
    return cropped_batch


def resize(image, image_dim):
    """
    Resize image to target dimensions.
    
    Args:
        image: Image tensor of shape [..., H, W, C]
        image_dim: Target (height, width)
        
    Returns:
        Resized image
    """
    assert len(image_dim) == 2
    
    # Convert HWC to CHW for PyTorch
    original_shape = image.shape
    if len(original_shape) == 3:
        # Single image: H, W, C -> C, H, W
        image_chw = image.permute(-1, -3, -2)
    else:
        # Batch: ..., H, W, C -> ..., C, H, W
        image_chw = image.transpose(-1, -3).transpose(-1, -2)
    
    # Resize
    resized_chw = F.interpolate(image_chw, size=image_dim, mode='bilinear', align_corners=False)
    
    # Convert back to HWC format
    if len(original_shape) == 3:
        # C, H, W -> H, W, C
        resized = resized_chw.permute(-2, -1, -3)
    else:
        # ..., C, H, W -> ..., H, W, C
        resized = resized_chw.transpose(-1, -2).transpose(-1, -3)
    
    return resized


def _maybe_apply(apply_fn, inputs, apply_prob):
    """Apply function with given probability."""
    should_apply = torch.rand(1).item() <= apply_prob
    if should_apply:
        return apply_fn(inputs)
    else:
        return inputs


def _gaussian_blur_single_image(image, kernel_size, sigma):
    """Apply gaussian blur to a single image."""
    # Convert to CHW format for PyTorch operations
    if len(image.shape) == 3:  # H, W, C
        image_chw = image.permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W
        squeeze_batch = True
    else:  # Already has batch dim
        image_chw = image.permute(0, 3, 1, 2)  # B, C, H, W
        squeeze_batch = False
    
    # Create Gaussian kernel
    kernel_1d = torch.tensor([
        math.exp(-(x - kernel_size // 2)**2 / (2 * sigma**2))
        for x in range(kernel_size)
    ], dtype=image.dtype, device=image.device)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Apply separable convolution
    # Horizontal pass
    kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(image_chw.shape[1], 1, 1, 1)
    padding_h = kernel_size // 2
    blurred = F.conv2d(image_chw, kernel_h, padding=(0, padding_h), groups=image_chw.shape[1])
    
    # Vertical pass
    kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(image_chw.shape[1], 1, 1, 1)
    padding_v = kernel_size // 2
    blurred = F.conv2d(blurred, kernel_v, padding=(padding_v, 0), groups=image_chw.shape[1])
    
    # Convert back to HWC format
    if squeeze_batch:
        blurred = blurred.squeeze(0).permute(1, 2, 0)  # H, W, C
    else:
        blurred = blurred.permute(0, 2, 3, 1)  # B, H, W, C
    
    return blurred


def _random_gaussian_blur(image, *, kernel_size, sigma_min, sigma_max, apply_prob):
    """Apply random gaussian blur."""
    def _apply(image):
        sigma = torch.rand(1).item() * (sigma_max - sigma_min) + sigma_min
        return _gaussian_blur_single_image(image, kernel_size, sigma)
    
    return _maybe_apply(_apply, image, apply_prob)


def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV."""
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    rangec = maxc - minc
    
    # Saturation
    s = torch.where(maxc > 0, rangec / maxc, torch.zeros_like(maxc))
    
    # Hue
    rc = (maxc - r) / (rangec + 1e-7)
    gc = (maxc - g) / (rangec + 1e-7)
    bc = (maxc - b) / (rangec + 1e-7)
    
    h = torch.where(r == maxc, bc - gc,
                   torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = h / 6.0
    h = h % 1.0
    
    # Set hue to 0 where range is 0
    h = torch.where(rangec == 0, torch.zeros_like(h), h)
    
    return h, s, maxc


def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB."""
    h = h % 1.0
    i = torch.floor(h * 6.0).long()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    idx = i % 6
    
    r = torch.where(idx == 0, v,
                   torch.where(idx == 1, q,
                              torch.where(idx == 2, p,
                                         torch.where(idx == 3, p,
                                                    torch.where(idx == 4, t, v)))))
    
    g = torch.where(idx == 0, t,
                   torch.where(idx == 1, v,
                              torch.where(idx == 2, v,
                                         torch.where(idx == 3, q,
                                                    torch.where(idx == 4, p, p)))))
    
    b = torch.where(idx == 0, p,
                   torch.where(idx == 1, p,
                              torch.where(idx == 2, t,
                                         torch.where(idx == 3, v,
                                                    torch.where(idx == 4, v, q)))))
    
    return r, g, b


def adjust_brightness(rgb_tuple, delta):
    """Adjust brightness by adding delta to all channels."""
    return tuple(channel + delta for channel in rgb_tuple)


def adjust_contrast(image, factor):
    """Adjust contrast by scaling around the mean."""
    def _adjust_contrast_channel(channel):
        mean = torch.mean(channel, dim=(-2, -1), keepdim=True)
        return factor * (channel - mean) + mean
    
    return tuple(_adjust_contrast_channel(channel) for channel in image)


def adjust_saturation(h, s, v, factor):
    """Adjust saturation by scaling the S channel."""
    return h, torch.clamp(s * factor, 0.0, 1.0), v


def adjust_hue(h, s, v, delta):
    """Adjust hue by adding delta to the H channel."""
    return (h + delta) % 1.0, s, v


def _random_brightness(rgb_tuple, max_delta):
    """Apply random brightness adjustment."""
    delta = torch.rand(1).item() * 2 * max_delta - max_delta
    return adjust_brightness(rgb_tuple, delta)


def _random_contrast(rgb_tuple, max_delta):
    """Apply random contrast adjustment."""
    factor = torch.rand(1).item() * 2 * max_delta + (1 - max_delta)
    return adjust_contrast(rgb_tuple, factor)


def _random_saturation(rgb_tuple, max_delta):
    """Apply random saturation adjustment."""
    h, s, v = rgb_to_hsv(*rgb_tuple)
    factor = torch.rand(1).item() * 2 * max_delta + (1 - max_delta)
    return hsv_to_rgb(*adjust_saturation(h, s, v, factor))


def _random_hue(rgb_tuple, max_delta):
    """Apply random hue adjustment."""
    h, s, v = rgb_to_hsv(*rgb_tuple)
    delta = torch.rand(1).item() * 2 * max_delta - max_delta
    return hsv_to_rgb(*adjust_hue(h, s, v, delta))


def _to_grayscale(image):
    """Convert image to grayscale."""
    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device, dtype=image.dtype)
    grayscale = torch.sum(image * rgb_weights, dim=-1, keepdim=True)
    return grayscale.repeat(*([1] * (len(image.shape) - 1)), 3)


def color_transform(
    image,
    *,
    brightness,
    contrast,
    saturation,
    hue,
    to_grayscale_prob,
    color_jitter_prob,
    apply_prob,
    shuffle
):
    """Apply color jittering to a single image."""
    # Whether to apply transform at all
    should_apply = torch.rand(1).item() <= apply_prob
    if not should_apply:
        return image
    
    # Whether to apply grayscale
    should_apply_gs = torch.rand(1).item() <= to_grayscale_prob
    
    # Whether to apply color jittering
    should_apply_color = torch.rand(1).item() <= color_jitter_prob
    
    if should_apply_color:
        # Split image into RGB channels
        rgb_tuple = tuple(image[..., i:i+1] for i in range(3))
        
        # Define transformations
        transforms = []
        if brightness > 0:
            transforms.append(lambda x: _random_brightness(x, brightness))
        if contrast > 0:
            transforms.append(lambda x: _random_contrast(x, contrast))
        if saturation > 0:
            transforms.append(lambda x: _random_saturation(x, saturation))
        if hue > 0:
            transforms.append(lambda x: _random_hue(x, hue))
        
        # Apply transforms in random order if shuffle is True
        if shuffle and transforms:
            import random
            random.shuffle(transforms)
        
        # Apply each transform
        for transform in transforms:
            rgb_tuple = transform(rgb_tuple)
        
        # Recombine channels and clamp
        image = torch.cat(rgb_tuple, dim=-1)
        image = torch.clamp(image, 0.0, 1.0)
    
    # Apply grayscale if selected
    if should_apply_gs:
        image = _to_grayscale(image)
    
    return torch.clamp(image, 0.0, 1.0)


def random_flip(image):
    """Randomly flip image horizontally."""
    should_flip = torch.rand(1).item() <= 0.5
    if should_flip:
        return torch.flip(image, dims=[-2])  # Flip width dimension
    return image


def gaussian_blur(
    image, *, blur_divider=10.0, sigma_min=0.1, sigma_max=2.0, apply_prob=1.0
):
    """Apply gaussian blur to an image."""
    kernel_size = int(image.shape[-3] / blur_divider)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    return _random_gaussian_blur(
        image,
        kernel_size=kernel_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        apply_prob=apply_prob
    )


def solarize(image, *, threshold, apply_prob):
    """Solarize the image by inverting pixels below threshold."""
    def _apply(image):
        return torch.where(image < threshold, image, 1.0 - image)
    
    return _maybe_apply(_apply, image, apply_prob)


def batched_color_transform(
    images,
    *,
    brightness,
    contrast,
    saturation,
    hue,
    to_grayscale_prob,
    color_jitter_prob,
    apply_prob,
    shuffle,
    num_batch_dims: int = 1,
):
    """Apply color jittering to a batch of images."""
    original_shape = images.shape
    batch_size = 1
    for i in range(num_batch_dims):
        batch_size *= images.shape[i]
    
    images_flat = images.reshape(batch_size, *images.shape[num_batch_dims:])
    
    # Check if input is uint8 and convert to float32
    is_uint8 = images.dtype == torch.uint8
    if is_uint8:
        images_flat = images_flat.float() / 255.0
    
    # Apply color transform to each image
    transformed_list = []
    for i in range(batch_size):
        transformed = color_transform(
            images_flat[i],
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            to_grayscale_prob=to_grayscale_prob,
            color_jitter_prob=color_jitter_prob,
            apply_prob=apply_prob,
            shuffle=shuffle
        )
        transformed_list.append(transformed)
    
    # Stack and convert back if needed
    transformed_batch = torch.stack(transformed_list, dim=0)
    
    if is_uint8:
        transformed_batch = (transformed_batch * 255.0).clamp(0, 255).byte()
    
    # Reshape back to original batch dimensions
    transformed_batch = transformed_batch.reshape(original_shape)
    
    return transformed_batch
