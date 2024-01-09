import cv2
import numpy as np
def calculate_turniness(contour):
    if len(contour) < 3:
        return 0  # Contour should have at least 3 points for angles to be calculated

    angles = []
    for i in range(len(contour) - 2):
        pt1 = contour[i][0]
        pt2 = contour[i + 1][0]
        pt3 = contour[i + 2][0]

        angle = np.degrees(np.arctan2(pt3[1] - pt2[1], pt3[0] - pt2[0]) - np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0]))
        angle = (angle + 360) % 360  # Ensure the angle is in the range [0, 360]
        angles.append(angle)

    return np.mean(angles)
def display_only_orange(image):
    notes = []
    # Load the image
    # image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for orange color in HSV
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([30, 255, 255])

    # Create a mask to extract only orange pixels
    orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    kernel = np.ones((5, 5), np.uint8)
    # orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(image)
    for i, contour in enumerate(contours):
        # Filter based on contour area (adjust the threshold as needed)
        area = cv2.contourArea(contour)
        if area > 1000 and len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
            circularity = calculate_turniness(contour)
            if circularity  > 180:
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                cv2.ellipse(contour_image, ellipse, (255, 0, 0), 2)
                cv2.drawContours(contour_image, contours, i, (255, 255, 0), 2)
                notes.append(ellipse)
                continue
            cv2.drawContours(contour_image, contours, i, (0, 255, 0), 2)
        else:
            cv2.drawContours(orange_mask, [contour], 0, 0, -1)
    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=orange_mask)

    for note in notes:
        center = (int(note[0][0]), int(note[0][1]))
        cv2.ellipse(result_image, note, (255, 0, 0), 2)
        cv2.circle(result_image, center, 5, (0, 255, 0), -1)  

    # Display the result
    return result_image, contour_image

# if __name__ == "__main__":
#     # Specify the path to your image
#     image_path = 'note2.png'

#     # Call the function to display only orange
#     img1, contour1 = display_only_orange(image_path)
#     img2, contour2 = display_only_orange('note.jpg')
#     img3, contour3 = display_only_orange('note3.png')
#     img1_scaled = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
#     img2_scaled = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
#     img3_scaled = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
#     cv2.imshow('Noisy Image', img1_scaled)
#     cv2.imshow('Clear Image', img2_scaled)
#     cv2.imshow('Overlap', img3_scaled)
    
#     cv2.imshow('Noisy Image 1', contour1)
#     cv2.imshow('Clear Image 1', contour2)
#     cv2.imshow('Overlap 1', contour3)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
camera = cv2.VideoCapture(1)
while True:
    ret, frame = camera.read()
    processed_frame, contour_map = display_only_orange(frame)
    cv2.imshow("Camera Stream", processed_frame)
    cv2.imshow("Camera", frame)
    cv2.imshow("Objects", contour_map)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("finished")
camera.release()
cv2.destroyAllWindows()