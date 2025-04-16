# Import necessary libraries
import cv2
import matplotlib.pyplot as plt

# Define the function (already provided above)
def generate_histogram_equalization(image_path, output_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)

    # Plot the original and equalized images with their histograms
    plt.figure(figsize=(12, 6))

    # Original Image and Histogram
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.hist(image.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title('Histogram of Original Image')

    # Equalized Image and Histogram
    plt.subplot(2, 2, 3)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.hist(equalized_image.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)
    plt.title('Histogram of Equalized Image')

    # Save the plot
    plt.savefig(output_path)

# Input image path
image_path = "35.jpg"

# Output plot path
output_path = "histogram_equalization.png"

# Call the function
generate_histogram_equalization(image_path, output_path)

print(f"Histogram equalization plot saved at: {output_path}")
