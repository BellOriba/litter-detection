import cv2
import numpy as np

# Load the image
image = cv2.imread('datasets\TACO\data\\batch_2\\000024.JPG')

# Resize the image while maintaining aspect ratio
max_dimension = 800  # Set a maximum size for the largest dimension
height, width = image.shape[:2]

# Calculate scaling factor
scaling_factor = max_dimension / float(max(height, width))

# Resize the image with the calculated scaling factor
new_width = int(width * scaling_factor)
new_height = int(height * scaling_factor)
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Convert to HSV color space for color filtering
hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# Define color range for filtering (example: bright colors)
lower_color = np.array([0, 50, 50])
upper_color = np.array([50, 255, 255])
mask = cv2.inRange(hsv, lower_color, upper_color)

# Apply Gaussian Blur to the mask
blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

# Use adaptive thresholding
thresholded = cv2.adaptiveThreshold(blurred_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

# Apply morphological operations
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove small noise
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)    # Close gaps

# Find contours
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around filtered contours
for contour in contours:
    # Filter based on contour area
    area = cv2.contourArea(contour)
    if 500 < area < 5000:  # Adjust area limits as needed
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 2.0:  # Filter based on aspect ratio
            cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the final image with detected litter
cv2.imshow('Enhanced Litter Detection', blurred_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
