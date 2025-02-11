import cv2
import numpy as np

def detectLine(frame):
    """
    Process the given frame to detect and track the center of a white line.
    
    Args:
        frame (numpy.ndarray): The input frame from the webcam.
    
    Returns:
        lineCenter: A number between [-1, 1] denoting where the center of the line is relative to the frame.
        newFrame: Processed frame with the detected line marked using cv2.rectangle() and center marked using cv2.circle().
    """

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to isolate the white line (adjust threshold as needed)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original frame and 0 (center)
    if len(contours) == 0:
        return 0, frame  

    # Find the largest contour (assumed to be the white line)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box around the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the center of the detected white line
    lineCenterX = x + w // 2  
    frameCenterX = frame.shape[1] // 2  

    # Normalize the line position between -1 (left) and 1 (right)
    normalizedCenter = (lineCenterX - frameCenterX) / (frame.shape[1] // 2)

    # Draw a bounding box around the detected line
    newFrame = frame.copy()
    cv2.rectangle(newFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw a circle at the center of the detected line
    cv2.circle(newFrame, (lineCenterX, y + h // 2), 5, (0, 0, 255), -1)

    return normalizedCenter, newFrame
def main():
    cam = cv2.VideoCapture(0)  # Open webcam

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        lineCenter, newFrame = detectLine(frame)

        cv2.imshow(newFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
