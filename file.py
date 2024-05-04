import numpy as np
import cv2
from skimage.transform import resize
from skimage.io import imread

def rgb2gray(rgb):
    """Convert RGB image to grayscale."""
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def compute_gradients(image):
    """Compute gradient magnitudes and orientations using Sobel filters."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = cv2.filter2D(image, -1, sobel_x)
    grad_y = cv2.filter2D(image, -1, sobel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    
    return magnitude, orientation

def calculate_histogram(orientation, magnitude, num_bins=9):
    """Calculate histogram of oriented gradients."""
    histogram = np.zeros(num_bins)
    bin_width = 180 / num_bins

    for i in range(num_bins):
        # Calculate histogram bin boundaries
        bin_start = i * bin_width
        bin_end = bin_start + bin_width

        # Select gradients within this bin range
        bin_gradients = magnitude[(orientation >= bin_start) & (orientation < bin_end)]

        # Weighted contribution to histogram bin
        histogram[i] = np.sum(bin_gradients)
    
    return histogram

def hog_descriptor(image, visualize=True, cell_size=(8, 8), block_size=(2, 2), num_bins=9):
    """Compute HOG descriptor for the entire image."""
    # Convert image to grayscale
    gray_image = rgb2gray(image)
    
    # Compute gradients
    magnitude, orientation = compute_gradients(gray_image)
    
    # Define cell and block sizes
    cell_height, cell_width = cell_size
    block_height, block_width = block_size
    
    # Calculate the number of cells and blocks
    num_cells_y = gray_image.shape[0] // cell_height
    num_cells_x = gray_image.shape[1] // cell_width
    num_blocks_y = num_cells_y - block_height + 1
    num_blocks_x = num_cells_x - block_width + 1
    
    # Initialize HOG descriptor
    hog_features = []
    
    # Initialize visualization image
    if visualize:
        vis_image = cv2.cvtColor(gray_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Iterate over all blocks
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            block_descriptor = []
            # Iterate over cells in each block
            for i in range(block_height):
                for j in range(block_width):
                    cell_y = y + i
                    cell_x = x + j
                    cell_orientation = orientation[cell_y * cell_height:(cell_y + 1) * cell_height,
                                                    cell_x * cell_width:(cell_x + 1) * cell_width]
                    cell_magnitude = magnitude[cell_y * cell_height:(cell_y + 1) * cell_height,
                                                cell_x * cell_width:(cell_x + 1) * cell_width]
                    histogram = calculate_histogram(cell_orientation, cell_magnitude, num_bins)
                    block_descriptor.extend(histogram)
                    # Visualize gradient orientation
                    if visualize:
                        center = ((cell_x + 0.5) * cell_width, (cell_y + 0.5) * cell_height)
                        magnitude_val = np.mean(cell_magnitude)
                        endpoint = (int(center[0] + magnitude_val * np.cos(np.deg2rad(np.mean(cell_orientation)))),
                                    int(center[1] + magnitude_val * np.sin(np.deg2rad(np.mean(cell_orientation)))))
                        cv2.line(vis_image, (int(center[0]), int(center[1])), endpoint, (0, 255, 0), 1)
            # Normalize block descriptor
            block_descriptor /= np.sqrt(np.sum(np.array(block_descriptor) ** 2) + 1e-5)
            hog_features.extend(block_descriptor)
    
    if visualize:
        return hog_features, vis_image
    else:
        return hog_features

# Example usage
image = imread('image1.jpg')
hog_features, hog_image = hog_descriptor(image)
print("HOG Features Shape:", np.array(hog_features).shape)

# Display image with HOG features
# cv2.imshow('HOG Features', hog_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
