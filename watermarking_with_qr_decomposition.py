# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 17:52:34 2024

@author: sau
"""

import numpy as np
from scipy.linalg import qr
import cv2
import hashlib
import secrets
import string
from skimage.metrics import structural_similarity as ssim
from ImageAttacks import ImageAttacks
import time

def ncc_color_images(img1, img2):
    # Check if images are of the same size
    if img1.shape != img2.shape:
        raise ValueError("Both images must have the same dimensions.")
    
    # Split color channels (B, G, R)
    channels1 = cv2.split(img1)
    channels2 = cv2.split(img2)

    # Compute NCC for each channel
    ncc_values = []
    for c1, c2 in zip(channels1, channels2):
        # Use template matching to calculate NCC
        result = cv2.matchTemplate(c1, c2, method=cv2.TM_CCORR_NORMED)
        ncc = result[0][0]  # Single value as both images are of the same size
        ncc_values.append(ncc)

    # Return the average NCC across all channels
    ncc_value = np.mean(ncc_values)
    return ncc_value

def key_based_arnold_iterations(key, min_iterations=1, max_iterations=10):
    """
    Generate a number of Arnold transform iterations based on a key.
    The key is hashed using MD5 and mapped to the range [min_iterations, max_iterations].
    """
    hash_digest = hashlib.md5(key.encode('utf-8')).hexdigest()  # Create an MD5 hash of the key
    hash_int = int(hash_digest, 16)  # Convert the hash to an integer
    iterations = min_iterations + (hash_int % (max_iterations - min_iterations + 1))  # Map to range
    return iterations

def generate_secure_key(length=16):
    """
    Generate a random secure key of specified length containing letters, digits, and symbols.
    """
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(characters) for _ in range(length))

def arnold_transform(image, iterations):
    """
    Apply the Arnold Transform iteratively to scramble an image.
    """
    n = image.shape[0]  # Assuming square images
    transformed = np.copy(image)
    for _ in range(iterations):
        temp = np.zeros_like(transformed)
        for i in range(n):
            for j in range(n):
                x = (i + j) % n
                y = (i + 2 * j) % n
                temp[x, y] = transformed[i, j]
        transformed = temp
    return transformed

def inverse_arnold_transform(image, iterations):
    """
    Apply the inverse Arnold Transform to restore the original image structure.
    """
    n = image.shape[0]  # Assuming square images
    transformed = np.copy(image)
    for _ in range(iterations):
        temp = np.zeros_like(transformed)
        for x in range(n):
            for y in range(n):
                i = (2 * x - y) % n  # Inverse coordinate transformation
                j = (-x + y) % n
                temp[i, j] = transformed[x, y]
        transformed = temp
    return transformed

def quantize_and_embed(block, watermark_bit, delta):
    """
    Embed a watermark bit into the first row, fourth column element of R matrix from QR decomposition.
    """
    block_matrix, position = block  # Extract the block matrix and its position
    Q, R = qr(block_matrix)  # Perform QR decomposition
    r_14 = R[0, 3]  # Get the (0, 3) element of the R matrix
    
    # Define thresholds based on the watermark bit
    if watermark_bit == '1':
        T1, T2 = 0.5 * delta, -1.5 * delta
    elif watermark_bit == '0':
        T1, T2 = -0.5 * delta, 1.5 * delta
    
    # Calculate candidate values
    k = int(np.floor(np.ceil(r_14 / delta) / 2))
    C1 = 2 * k * delta + T1
    C2 = 2 * k * delta + T2
    
    # Update r_14 to the nearest candidate value
    if abs(r_14 - C2) < abs(r_14 - C1):
        R[0, 3] = C2
    else:
        R[0, 3] = C1
    
    # Reconstruct the block with the updated R matrix
    watermarked_block_matrix = np.dot(Q, R)
    watermarked_block = (watermarked_block_matrix, position)  # Return updated block and position
    return watermarked_block

def split_into_channels(img):
    """
    Split an RGB image into its red, green, and blue channels.
    """
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B

def blockify(channel, block_size=4):
    """
    Divide an image channel into non-overlapping blocks of size block_size x block_size.
    """
    H, W = channel.shape  # Get channel dimensions
    vertical_blocks = H // block_size
    horizontal_blocks = W // block_size
    
    blocks = []
    for v in range(vertical_blocks):
        for h in range(horizontal_blocks):
            # Extract each block and record its position
            block = channel[v*block_size:(v+1)*block_size, h*block_size:(h+1)*block_size]
            position = (v, h)
            blocks.append((block.copy(), position))
    return blocks

def md5_based_selection(blocks, select_count, seed_str="default_seed"):
    """
    Select a specific number of blocks using MD5 hash-based indexing.
    """
    md5_hash = hashlib.md5(seed_str.encode('utf-8')).hexdigest()  # Create an MD5 hash of the seed string
    hash_int = int(md5_hash, 16)  # Convert the hash to an integer

    N = len(blocks)  # Total number of blocks
    selected_blocks = []
    current_val = hash_int  # Initialize with the hash value
    
    for _ in range(select_count):
        idx = current_val % N  # Use modulus to wrap around the index
        selected_blocks.append(blocks[idx])  # Append the block at the computed index
        # Update the current value using a linear congruential generator formula
        current_val = (current_val * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
    
    return selected_blocks

def reconstruct_channel_from_selected_blocks(selected_blocks, block_size, height, width):
    """
    Reconstruct a single image channel using selected blocks.
    Unselected regions are initialized to zero.
    """
    reconstructed = np.zeros((height, width), dtype=selected_blocks[0][0].dtype)  # Initialize with zeros
    
    for block, (v, h) in selected_blocks:
        row_start = v * block_size
        col_start = h * block_size
        # Copy the block into its original position in the reconstructed channel
        reconstructed[row_start:row_start+block_size, col_start:col_start+block_size] = block
        
    return reconstructed

def reconstruct_full_image(selected_R_blocks, selected_G_blocks, selected_B_blocks, block_size, height, width):
    """
    Combine reconstructed R, G, and B channels into a full RGB image.
    """
    R_reconstructed = reconstruct_channel_from_selected_blocks(selected_R_blocks, block_size, height, width)
    G_reconstructed = reconstruct_channel_from_selected_blocks(selected_G_blocks, block_size, height, width)
    B_reconstructed = reconstruct_channel_from_selected_blocks(selected_B_blocks, block_size, height, width)
    
    # Stack the channels to create the final RGB image
    img_reconstructed = np.stack([R_reconstructed, G_reconstructed, B_reconstructed], axis=2)
    return img_reconstructed

def channels_to_bitstrings(R, G, B):
    """
    Convert pixel values from R, G, and B channels into binary bitstrings.
    """
    def channel_to_bitstring(channel):
        # Flatten the channel and convert each pixel to an 8-bit binary string
        flattened = channel.ravel()
        bit_string = "".join([format(val, '08b') for val in flattened])
        return bit_string
    
    R_bits = channel_to_bitstring(R)
    G_bits = channel_to_bitstring(G)
    B_bits = channel_to_bitstring(B)

    return R_bits, G_bits, B_bits

def bitstrings_to_image(R_bits, G_bits, B_bits, height, width):
    """
    Convert R, G, and B bitstrings back into pixel values and reconstruct the original image.
    """
    N = height * width  # Total number of pixels

    # Ensure the bitstrings are valid
    assert len(R_bits) == N * 8, "R_bits length does not match pixel count!"
    assert len(G_bits) == N * 8, "G_bits length does not match pixel count!"
    assert len(B_bits) == N * 8, "B_bits length does not match pixel count!"
    
    def bits_to_channel(channel_bits):
        # Parse 8-bit segments into pixel values and reshape into the original dimensions
        pixels = [int(channel_bits[i:i+8], 2) for i in range(0, len(channel_bits), 8)]
        return np.array(pixels, dtype=np.uint8).reshape(height, width)
    
    R = bits_to_channel(R_bits)
    G = bits_to_channel(G_bits)
    B = bits_to_channel(B_bits)
    
    # Combine the channels to reconstruct the image
    img = np.stack([R, G, B], axis=2)
    return img

def merge_blocks(original_blocks, selected_blocks):
    """
    Merge original blocks with selected blocks, replacing originals where necessary.
    """
    # Store original blocks in a dictionary indexed by position
    blocks_dict = {position: block_matrix for block_matrix, position in original_blocks}
    
    # Update the dictionary with selected blocks
    for block_matrix, position in selected_blocks:
        blocks_dict[position] = block_matrix
    
    # Convert the dictionary back to a sorted list of blocks
    merged_blocks = sorted(blocks_dict.items(), key=lambda x: x[0])  # Sort by position
    merged_blocks = [(block_matrix, position) for position, block_matrix in merged_blocks]
    
    return merged_blocks

def embed_watermark(host_image, watermark, delta):
    """
    Embed a watermark into a host image using Arnold Transform, block selection, and quantization.
    """
    host_image = host_image.astype(float)  # Convert to float for processing
    
    # Generate secure keys for selection and Arnold Transform
    KA = generate_secure_key(length=16)
    iterations = key_based_arnold_iterations(KA, min_iterations=1, max_iterations=10)

    # Apply Arnold Transform to the watermark
    watermark_r, watermark_g, watermark_b = split_into_channels(watermark)
    watermark_r = arnold_transform(watermark_r, iterations)
    watermark_g = arnold_transform(watermark_g, iterations)
    watermark_b = arnold_transform(watermark_b, iterations)

    # Split host image into R, G, B channels
    host_r, host_g, host_b = split_into_channels(host_image)

    # Divide each channel into blocks
    R_blocks = blockify(host_r, block_size=4)
    G_blocks = blockify(host_g, block_size=4)
    B_blocks = blockify(host_b, block_size=4)

    # Determine the number of watermark bits to embed
    block_sizes = watermark_r.shape[0] * watermark_r.shape[1] * 8  # Bits in watermark

    # Generate keys for block selection
    K_1 = generate_secure_key(length=16)
    K_2 = generate_secure_key(length=16)
    K_3 = generate_secure_key(length=16)
    K = (K_1, K_2, K_3)

    # Select blocks using MD5-based selection
    selected_R_blocks = md5_based_selection(R_blocks, select_count=block_sizes, seed_str=K_1)
    selected_G_blocks = md5_based_selection(G_blocks, select_count=block_sizes, seed_str=K_2)
    selected_B_blocks = md5_based_selection(B_blocks, select_count=block_sizes, seed_str=K_3)

    # Convert watermark channels to bitstrings
    R_bits, G_bits, B_bits = channels_to_bitstrings(watermark_r, watermark_g, watermark_b)

    # Embed watermark bits into selected blocks
    for i in range(block_sizes):
        selected_R_blocks[i] = quantize_and_embed(selected_R_blocks[i], R_bits[i], delta)
        selected_G_blocks[i] = quantize_and_embed(selected_G_blocks[i], G_bits[i], delta)
        selected_B_blocks[i] = quantize_and_embed(selected_B_blocks[i], B_bits[i], delta)

    # Merge modified blocks with original blocks
    merged_R_blocks = merge_blocks(R_blocks, selected_R_blocks)
    merged_G_blocks = merge_blocks(G_blocks, selected_G_blocks)
    merged_B_blocks = merge_blocks(B_blocks, selected_B_blocks)

    # Reconstruct the watermarked image
    watermarked_image = reconstruct_full_image(merged_R_blocks, merged_G_blocks, merged_B_blocks, 4, host_r.shape[0], host_r.shape[1])

    # Compute metrics (SSIM, PSNR) to evaluate embedding quality
    ssim_value = ssim(watermarked_image, host_image, channel_axis=2, data_range=255, win_size=3)
    psnr_value = cv2.PSNR(host_image, watermarked_image)

    return watermarked_image, K, KA, ssim_value, psnr_value

def extract_watermark(watermarked_image, K, KA, delta, watermark_shape):
    """
    Extract the watermark from the watermarked image using the keys and quantization values.
    """
    # Split the watermarked image into R, G, B channels
    watermarked_r, watermarked_g, watermarked_b = (
        watermarked_image[..., 0],
        watermarked_image[..., 1],
        watermarked_image[..., 2],
    )

    # Divide each channel into blocks
    R_blocks = blockify(watermarked_r, block_size=4)
    G_blocks = blockify(watermarked_g, block_size=4)
    B_blocks = blockify(watermarked_b, block_size=4)

    # Determine the number of blocks needed for the watermark
    block_sizes = watermark_shape * watermark_shape * 8

    # Retrieve keys for block selection
    K_1 = K[0]
    K_2 = K[1]
    K_3 = K[2]

    # Select the same blocks used during embedding
    selected_R_blocks = md5_based_selection(R_blocks, select_count=block_sizes, seed_str=K_1)
    selected_G_blocks = md5_based_selection(G_blocks, select_count=block_sizes, seed_str=K_2)
    selected_B_blocks = md5_based_selection(B_blocks, select_count=block_sizes, seed_str=K_3)

    # Initialize bitstrings for extracted watermark
    R_bits = ''
    G_bits = ''
    B_bits = ''

    # Extract bits from the R channel
    for block, position in selected_R_blocks:
        Q, R = qr(block)
        r_14 = R[0, 3]
        # Extract watermark bit using quantization step
        R_bits += str(np.mod(int(np.ceil(r_14 / delta)), 2))

    # Extract bits from the G channel
    for block, position in selected_G_blocks:
        Q, R = qr(block)
        r_14 = R[0, 3]
        G_bits += str(np.mod(int(np.ceil(r_14 / delta)), 2))

    # Extract bits from the B channel
    for block, position in selected_B_blocks:
        Q, R = qr(block)
        r_14 = R[0, 3]
        B_bits += str(np.mod(int(np.ceil(r_14 / delta)), 2))

    # Reconstruct the extracted watermark from the bitstrings
    extracted_watermark = bitstrings_to_image(R_bits, G_bits, B_bits, watermark_shape, watermark_shape)

    # Apply inverse Arnold Transform to each component to restore the watermark
    iterations = key_based_arnold_iterations(KA, min_iterations=1, max_iterations=10)
    extracted_watermark[:, :, 0] = inverse_arnold_transform(extracted_watermark[:, :, 0], iterations)
    extracted_watermark[:, :, 1] = inverse_arnold_transform(extracted_watermark[:, :, 1], iterations)
    extracted_watermark[:, :, 2] = inverse_arnold_transform(extracted_watermark[:, :, 2], iterations)

    return extracted_watermark

# Main Script
host_image = cv2.imread('test_images/airplane.png')  # Load host image
watermark = cv2.imread('test_images/peugeot.png')   # Load watermark image
cv2.imshow('Original image', host_image)  # Display the host image

# Embedding Process
print('*' * 20, 'Embedding', '*' * 20)
delta = 22  # Quantization step
start_time = time.time()
watermarked_image, K, KA, ssim_value, psnr_value = embed_watermark(host_image, watermark, delta)
end_time = time.time()
time_elapsed1 = end_time - start_time
watermarked_image = watermarked_image.astype(np.uint8)
print('SSIM:', ssim_value)
print('PSNR:', psnr_value)
print('Key K:', K)
print('Key KA:', KA)
print('Embedding Time:', time_elapsed1, 'seconds')

cv2.imshow('Watermarked image', watermarked_image)  # Display the watermarked image

# Attacks on the Watermarked Image
attacker = ImageAttacks(watermarked_image)
# Uncomment one of the following lines to apply an attack:
# watermarked_image = attacker.jpeg_compression(200)
# watermarked_image = attacker.scaling(0.75)
# watermarked_image = attacker.cropping_left()
# watermarked_image = attacker.cropping_top_left()
# watermarked_image = attacker.brighten(40)
# watermarked_image = attacker.darken(-40)
# watermarked_image = attacker.gaussian_noise(0, 1)
# watermarked_image = attacker.salt_pepper_noise(0.02)
# watermarked_image = attacker.poisson_noise()
# watermarked_image = attacker.motion_blur()
# watermarked_image = attacker.histogram_equalization()

cv2.imshow('Attacked Watermarked Image', watermarked_image)  # Display attacked watermarked image

# Extraction Process
print('*' * 20, 'Extraction', '*' * 20)
start_time = time.time()
water_mark_ = extract_watermark(watermarked_image, K, KA, delta, 32)
end_time = time.time()
time_elapsed2 = end_time - start_time
print('NC:', ncc_color_images(water_mark_, watermark))  # Compute normalized correlation with the original watermark
print('Extraction Time:', time_elapsed2, 'seconds')
print('Total Time:', time_elapsed1 + time_elapsed2)

# Resize extracted watermark for visualization
# water_mark_ = cv2.resize(water_mark_, (128, 128))
cv2.imshow('Extracted Watermark', water_mark_)
cv2.waitKey(0)
cv2.destroyAllWindows()
