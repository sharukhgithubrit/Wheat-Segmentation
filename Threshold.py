import cv2 as cv
import numpy as np

image_path = "WHEAT-lower.png"
# Read the image using OpenCV
image = cv.imread(image_path)

def optimal_threshold(value):
    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blur = cv.GaussianBlur(gray_image, (5, 5), 0)

    # Apply the optimal threshold
    _, binary_image = cv.threshold(blur, value, 255, cv.THRESH_BINARY)

    # Display the original, blurred, and thresholded images
    cv.imshow('Original Image', image)
    cv.imshow('Blurred Image', blur)
    cv.imshow('Result', binary_image)

cv.namedWindow('Result')
cv.createTrackbar('Threshold', 'Result', 0, 255, optimal_threshold)
optimal_threshold(105)  # Initialize with the default threshold (you can change it)

cv.waitKey(0)
cv.destroyAllWindows()
