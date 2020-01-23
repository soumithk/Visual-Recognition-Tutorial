# Import necessary libraries
import cv2
import numpy as np


#####################################################
# One possible solution to exercise described here
# https://github.com/soumithk/Visual-Recognition-Tutorial/blob/master/opencv-tutorial1/opencv-part1.ipynb
# To run 'python exercise1_solution.py'
# cv2 version used to build the code : 3.1.0
# This code simultaneously displays the webcam stream
# and also writes it into a video
# press 'q' to exit the display window
#####################################################

# returns image with color filled
def draw_box(input_img, box_coords) :

    # CASE : no box is found in the image
    if len(box_coords) == 0 : return input_img

    image = input_img.copy()
    # color the box
    cv2.fillPoly(image, pts =[box_coords], color=(0,255,255))
    # draw border
    cv2.drawContours(image, [box_coords], -1, (0, 255, 0), 3)
    return image

# returns box coords
def find_box(input_img) :

    # Apply canny and find contours
    gray = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    # Sort the contours and keep top three
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0:3]

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Box shaped countour found : return
        if len(approx) == 4:
            return approx
    return []


if __name__ == "__main__" :

    # Open the stream
    cap = cv2.VideoCapture(0)


    # check
    if cap.isOpened() :
        # read the first frame
        ret, frame = cap.read()
        # Create a video writer object to write the video
        out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1], frame.shape[0]))
    else :
        ret = False
        print("Unable to open camera")


    while ret :

        # find the box coordinates and fill the box
        box_coords = find_box(frame)
        frame = draw_box(frame, box_coords)

        # display
        cv2.imshow("preview", frame)
        # write into the video
        out.write(frame)

        # exit on pressing 'q'
        if cv2.waitKey(10) == ord('q'):
            break

        # skip 2 frames. Adjust based on frame rate
        for i in range(3):
            ret, frame = cap.read()

    # cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
