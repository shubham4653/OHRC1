import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap(image_path, output_path):
    """
    Generate a heatmap based on shadow intensity (pixel values) of an input image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the generated heatmap.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Normalize pixel values to range [0, 1] (shadow intensity)
    shadow_intensity = image / 255.0

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(shadow_intensity, cmap="YlGnBu", cbar=True)
    plt.title("Shadow Intensity Heatmap")

    # Save the heatmap
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory

    print(f"Heatmap saved at: {output_path}")


# Input image path (predefined for now)
image_path = "35.jpg"  # Replace with your actual image path

# Output heatmap path
output_path = "heatmap_shadow_intensity.png"

# Call the function to generate the heatmap
generate_heatmap(image_path, output_path)

print(f"Heatmap saved at: {output_path}")
