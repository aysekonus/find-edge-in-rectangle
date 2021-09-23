import cv2
import numpy as np
import math

def approxSplit(approx,o=False):
    _arrayTmp0=str(approx[0])
    _arrayTmp0 = _arrayTmp0.replace("[", "")
    _arrayTmp0 = _arrayTmp0.replace("]", "")
    _array_1 = _arrayTmp0.split(" ")
    _newArray = []
    for x in _array_1:
        if(x.isnumeric()):
            _newArray.append(x)
    return _newArray[0],_newArray[1]

def drawingCenters(frame,x1,y1,x2,y2):
    cv2.line(frame,(x1,y1),(x2,y2),(255,255,0))

def findToCoordinatePoint(x,y):
    x=x-320
    y=190-y
    return x,y

def nothing(x):
    # any operation
    pass

while True:
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
        cv2.createTrackbar("L-S", "Trackbars", 66, 255, nothing)
        cv2.createTrackbar("L-V", "Trackbars", 134, 255, nothing)
        cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
        cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)

        font = cv2.FONT_HERSHEY_COMPLEX

        frame = cv2.imread(r'C:\Users\Lenovo\Desktop\images\sualti2.jpeg')
        frame = cv2.resize(frame, (555,480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L-H", "Trackbars")
        l_s = cv2.getTrackbarPos("L-S", "Trackbars")
        l_v = cv2.getTrackbarPos("L-V", "Trackbars")
        u_h = cv2.getTrackbarPos("U-H", "Trackbars")
        u_s = cv2.getTrackbarPos("U-S", "Trackbars")
        u_v = cv2.getTrackbarPos("U-V", "Trackbars")


        lower_yellow = np.array([l_h, l_s, l_v])
        upper_yellow = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(mask,kernel,iterations=1)
        erode = cv2.erode(dilate, kernel, iterations=1)


        contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if area>5000:
                if len(approx) == 4:
                    cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                    x1,y1=approxSplit(approx[0])
                    x2,y2=approxSplit(approx[2])
                    findToCoordinatePoint(x, y)
                    cv2.circle(frame, (int(x1),int(y1)), 5, (0,255,100))
                    cv2.circle(frame, (int(x2),int(y2)), 5, (0,255,100))
                    cv2.circle(frame, (int(x2),int(y1)), 5, (0,255,100))
                    cv2.circle(frame, (int(x1),int(y2)), 5, (0,255,100))

                    a = float((int(x2)-int(x1))*(int(x2)-int(x1)) + (int(y2)-int(y1))*(int(y2)-int(y1)))
                    b = float((int(y2)-int(y1))*(int(y2)-int(y1)))
                    a1 = int(math.sqrt(a))
                    b1 = int(math.sqrt(b))
                    # cv2.putText(frame, f"{a1},{b1}",(50,100), cv2.FONT_HERSHEY_SIMPLEX,1, (0,222,222))


                    # a2 = ((int(x2)-int(x1))^2 + (int(y2)-int(y1))^2)^(1/2)
                    # b2 = ((int(x2)-int(x1))^2 + (int(y2)-int(y1))^2)^(1/2)

                    w=int(float((int(x2)-int(x1))/2))
                    h=int(float((int(y2)-int(y1))/2 ))
                    ort1=int(x1)+w
                    ort2=int(y1)+h
                    # drawingCenters(frame, 320, 190, ort1, ort2)
                    # cv2.circle(frame, (ort1,ort2), 5, (0,255,100))
                    ort1,ort2=findToCoordinatePoint(ort1, ort2)
                    cv2.putText(frame, f"x:{ort1},y:{ort2}",(500,100), cv2.FONT_HERSHEY_SIMPLEX,1, (0,222,222))
                    #goToTheTarget(ort1, ort2)

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        
        oran = a1/b1
        print(a1,b1,oran)
        

cv2.waitKey(0)