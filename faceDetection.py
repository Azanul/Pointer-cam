import cv2




cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame1 = cap.read()
    if frame1 is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame1)
    if cv2.waitKey(10) == 27:
        break