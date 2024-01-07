import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Read the image
image_path = "WHEAT-lower.png"
image = cv.imread(image_path)

# Convert the image to grayscale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Display the original grayscale image
cv.imshow('Original Image',image)
cv.waitKey(0)
cv.destroyAllWindows()


# Calculate and plot the histogram of pixel values
histogram = cv.calcHist([gray_image], [0], None, [256], [0, 256])
plt.plot(histogram)
plt.title("Histogram for WHEAT data")
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Apply Gaussian blur to the grayscale image
blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)

# Use adaptive thresholding to create a binary mask
_, binary_mask = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Morphological operations to clean up the binary mask
kernel = np.ones((5, 5), np.uint8)
binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, kernel)
binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel)

# Apply the binary mask to the original colored image
result_image = cv.bitwise_and(image, image, mask=binary_mask)

# Display the resultant images
cv.imshow('Resultant Image', result_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the resultant image
cv.imwrite("Result_Image.png", result_image)

