import cv2
import numpy as np

def display_only_orange(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for orange color in HSV
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([30, 255, 255])

    # Create a mask to extract only orange pixels
    orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    kernel = np.ones((5, 5), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:  # Adjust threshold as needed
            cv2.drawContours(orange_mask, [contour], 0, 0, -1)  # Fill small contours with black
    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=orange_mask)
    gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance features
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=20, minRadius=50, maxRadius=1000
    )

    # If circles are found, draw them on the image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(result_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(result_image, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        print("No circles found")

    # Display the result
    return result_image

if __name__ == "__main__":
    # Specify the path to your image
    image_path = 'note2.png'

    # Call the function to display only orange
    img1 = display_only_orange(image_path)
    img2 = display_only_orange('note.jpg')
    img1_scaled = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    img2_scaled = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Noisy Image', img1_scaled)
    cv2.imshow('Clear Image', img2_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
