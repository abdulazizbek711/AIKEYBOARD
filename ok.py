import cv2

cap=cv2.VideoCapture(0)
cap.set(3, 1288)
cap.set(4, 728)
while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)