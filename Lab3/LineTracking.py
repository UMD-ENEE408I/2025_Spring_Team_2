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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    blur_gray = gray # cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    ret, thresh = cv2.threshold(blur_gray, 200, 255, cv2.THRESH_BINARY)

    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel)


    edges = cv2.Canny(thresh, 150, 50)
    rho = 1

    theta = np.pi * 2/180
    threshold = 50
    min_line_len = 100
    max_line_gap = 10
    line_image = np.copy(frame)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_len, max_line_gap)
    center = None
    str_lines = []
    if lines is not None:   
        for line in lines:
            for x1,y1,x2,y2 in line:
                if x1 > 480 or x2 > 480 or x1 < 160 or x2 < 160:
                    pass
                else:
                    if abs(x1 - x2) < 30:

                        str_lines.append(line[0])
                    # cv2.circle(line_image, center, radius=1, color=(255, 0, 0), thickness=-1)

                    # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    if len(str_lines) < 2:
        print("not enough vertical lines")
    if len(str_lines) >= 2:
        all_x = []
        all_y = []
        for (x1, y1, x2, y2) in str_lines:
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        center = (int((min_x+max_x)/2), int((min_y + max_y)/2))
        cv2.circle(line_image, center, radius=1, color=(255, 0, 0), thickness=-1)

        cv2.rectangle(line_image,(min_x,min_y),(max_x,max_y),(255,0,0),5)
    return center, line_image


def main():
    cam = cv2.VideoCapture(1)  # Open webcam

    while cam.isOpened():
        ret, frame = cam.read()

        if not ret:
            break

        lineCenter, newFrame = detectLine(frame)

        cv2.imshow("new frame", newFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
