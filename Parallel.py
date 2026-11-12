
# ============================================================
# PARALLEL CELLULAR ALGORITHM (PCA)
# APPLICATION: IMAGE DENOISING
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte

# ------------------------------------------------------------
# STEP 1: LOAD ORIGINAL IMAGE
# ------------------------------------------------------------

# Load sample grayscale image
original = data.camera()

# Convert image to floating-point values (0 to 255)
original = original.astype(np.float64)

print("Original Image Loaded Successfully!")
print("Image Dimensions:", original.shape)


# ------------------------------------------------------------
# STEP 2: ADD SALT AND PEPPER NOISE
# ------------------------------------------------------------

def add_noise(image, noise_amount=0.08):

    noisy = image.copy()

    rows, cols = image.shape

    # Generate random positions for noise
    total_pixels = rows * cols
    number_of_noise_pixels = int(noise_amount * total_pixels)

    rng = np.random.default_rng(42)

    positions = rng.choice(
        total_pixels,
        number_of_noise_pixels,
        replace=False
    )

    # Half pixels become black (Pepper)
    # Half pixels become white (Salt)
    half = number_of_noise_pixels // 2

    flat = noisy.flatten()

    flat[positions[:half]] = 0
    flat[positions[half:]] = 255

    return flat.reshape(image.shape)


noisy_image = add_noise(original)

print("Salt and Pepper Noise Added Successfully!")


# ------------------------------------------------------------
# STEP 3: PARALLEL CELLULAR ALGORITHM
# ------------------------------------------------------------

def parallel_cellular_algorithm(image, iterations=5):

    current = image.copy()

    psnr_values = []
    mse_values = []

    for iteration in range(iterations):

        # Pad image to handle boundary pixels
        padded = np.pad(current, 1, mode='reflect')

        # Each cell observes its 3x3 neighborhood
        # All neighborhoods are processed simultaneously

        neighbors = np.stack([
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:]
        ], axis=0)

        # Calculate median value of neighboring cells
        median_values = np.median(neighbors, axis=0)

        # Calculate difference between cell and median
        difference = np.abs(current - median_values)

        # Cellular update rule:
        # Update cells that appear to be noisy
        noisy_cells = (
            (current <= 20) |
            (current >= 235) |
            (difference > 40)
        )

        # Simultaneous state updating
        # Every cell uses the previous iteration's values
        updated = np.where(
            noisy_cells,
            median_values,
            current
        )

        current = updated

        # Evaluate image quality
        mse = np.mean((original - current) ** 2)

        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10((255 ** 2) / mse)

        mse_values.append(mse)
        psnr_values.append(psnr)

        print(
            f"Iteration {iteration + 1}: "
            f"MSE = {mse:.2f}, PSNR = {psnr:.2f} dB"
        )

    return current, mse_values, psnr_values


# ------------------------------------------------------------
# STEP 4: EXECUTE PCA
# ------------------------------------------------------------

print("\nExecuting Parallel Cellular Algorithm...\n")

denoised_image, mse_values, psnr_values = (
    parallel_cellular_algorithm(
        noisy_image,
        iterations=5
    )
)

print("\nImage Denoising Completed Successfully!")


# ------------------------------------------------------------
# STEP 5: CALCULATE IMAGE QUALITY
# ------------------------------------------------------------

def calculate_mse(original, processed):

    return np.mean((original - processed) ** 2)


def calculate_psnr(original, processed):

    mse = calculate_mse(original, processed)

    if mse == 0:
        return float('inf')

    return 10 * np.log10((255 ** 2) / mse)


noisy_mse = calculate_mse(original, noisy_image)
denoised_mse = calculate_mse(original, denoised_image)

noisy_psnr = calculate_psnr(original, noisy_image)
denoised_psnr = calculate_psnr(original, denoised_image)

print("\n========== FINAL RESULTS ==========")

print(f"Noisy Image MSE: {noisy_mse:.2f}")
print(f"Denoised Image MSE: {denoised_mse:.2f}")

print(f"Noisy Image PSNR: {noisy_psnr:.2f} dB")
print(f"Denoised Image PSNR: {denoised_psnr:.2f} dB")

print("===================================")


# ------------------------------------------------------------
# STEP 6: DISPLAY IMAGE COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(14, 9))

plt.subplot(2, 2, 1)
plt.imshow(original, cmap='gray', vmin=0, vmax=255)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
plt.title("Image with Salt and Pepper Noise")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(denoised_image, cmap='gray', vmin=0, vmax=255)
plt.title("PCA Denoised Image")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(range(1, 6), psnr_values, marker='o')
plt.title("PSNR Improvement Across Iterations")
plt.xlabel("Iterations")
plt.ylabel("PSNR (dB)")
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 7: FINAL PERFORMANCE SUMMARY
# ------------------------------------------------------------

print("\n========== PERFORMANCE SUMMARY ==========")

if denoised_mse < noisy_mse:
    print("Noise Reduction: SUCCESS")

if denoised_psnr > noisy_psnr:
    print("Image Quality: IMPROVED")

print("Total Cellular Iterations:", 5)
print("Neighborhood Structure: 3 x 3")
print("Number of Neighbors: 8")
print("Cell Updating: Parallel (Vectorized)")

print("==========================================")

