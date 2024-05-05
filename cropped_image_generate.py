import cv2
import os

# Global variables to store starting and ending (x, y) coordinates of the ROI
start_x, start_y, end_x, end_y = 0, 0, 0, 0
cropping = False

def mouse_event(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y, end_x, end_y = x, y, x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            end_x, end_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        end_x, end_y = x, y
        cropping = False
        # Ensure end_x and end_y are greater than start_x and start_y
        start_x, end_x = min(start_x, end_x), max(start_x, end_x)
        start_y, end_y = min(start_y, end_y), max(start_y, end_y)
        # Crop the selected region
        cropped_image = image[start_y:end_y, start_x:end_x]
        cv2.imshow("Cropped Image", cropped_image)
        cv2.waitKey(0)
        # Save the cropped image with the name "crop.jpg" in the same folder
        cv2.imwrite("crop.jpg", cropped_image)

# Read the image
image = cv2.imread("satellite.jpg")
clone = image.copy()

# Create a window and attach the mouse callback function
cv2.namedWindow("Input Image")
cv2.setMouseCallback("Input Image", mouse_event)

while True:
    cv2.imshow("Input Image", image)
    key = cv2.waitKey(1) & 0xFF

    # If the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break

# Close all windows
cv2.destroyAllWindows()