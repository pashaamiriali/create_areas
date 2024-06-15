import cv2
import numpy as np
import json

# Load the image
image = cv2.imread('map.png')
image_height, image_width, _ = image.shape

# Convert HEX to BGR
target_color_bgr = [216, 224, 231]

# Define the color range for the highlighted areas
lower_color = np.array([215, 223, 230])  # Lower bound of the color
upper_color = np.array([217, 225, 232])  # Upper bound of the color

# Create a mask to extract the areas with the defined color
mask = cv2.inRange(image, lower_color, upper_color)

# Detect white writings (white areas) within the red areas
lower_white = np.array([200, 200, 200])
upper_white = np.array([255, 255, 255])
white_mask = cv2.inRange(image, lower_white, upper_white)

# Combine the red mask with the white mask to treat white writings as red, but only inside red areas
white_in_red = cv2.bitwise_and(white_mask, mask)
combined_color_mask = cv2.bitwise_or(mask, white_in_red)

# Detect black lines using edge detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 50, 150)

# Dilate edges to make thin lines thicker
kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# Invert the dilated edges to combine with the mask
inverted_edges = cv2.bitwise_not(dilated_edges)

# Combine the color mask with the dilated edges to separate the areas
combined_mask = cv2.bitwise_and(combined_color_mask, inverted_edges)

# Apply morphology operations to remove small white noises from the mask
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

# Find contours (the boundaries of the color-coded areas)
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# List to store the relative polygons of the detected areas
relative_polygons = []

for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    relative_polygon = [(point[0][0] / image_width, point[0][1] / image_height) for point in approx]
    relative_polygons.append(relative_polygon)

# Save the relative polygons for later use in Flutter
with open('relative_polygons.json', 'w') as f:
    json.dump(relative_polygons, f)
