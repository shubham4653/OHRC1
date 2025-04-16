import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def generate_fourier_transform(image_path, output_path):
    """
    Generate an enhanced Fourier Transform magnitude spectrum plot for an input image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the generated Fourier Transform plot.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Compute the Fourier Transform
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)

    # Compute the magnitude spectrum
    magnitude_spectrum = np.log(1 + np.abs(f_shifted))

    # Define a custom colormap (blue → white → red)
    colors = ['blue', 'white', 'red']
    custom_cmap = LinearSegmentedColormap.from_list('blue_white_red', colors)

    # Plot the original image and its Fourier Transform
    plt.figure(figsize=(14, 7))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image", fontsize=14)
    plt.axis('off')

    # Fourier Transform Magnitude Spectrum (enhanced with colormap)
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap=custom_cmap)
    plt.colorbar(label='Magnitude Spectrum Intensity')  # Add colorbar for reference
    plt.title("Enhanced Fourier Transform Magnitude Spectrum", fontsize=14)
    plt.axis('off')

    # Save the plot
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory
    print(f"Fourier Transform plot saved at: {output_path}")


# Input image path (predefined for now)
image_path = "35.jpg"  # Replace with your actual image path

# Output Fourier Transform plot path
output_path = "enhanced_fourier_transform.png"

# Call the function to generate the enhanced Fourier Transform plot
generate_fourier_transform(image_path, output_path)

print(f"Enhanced Fourier Transform plot saved at: {output_path}")

