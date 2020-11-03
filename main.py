import cv2
import numpy as np


def thresh(img, lim, inv, label):
    # blur = cv2.medianBlur(img, 15)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    thresh_type = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    th = cv2.inRange(blur, lim[0], lim[1], thresh_type)
    img = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow(label, img)
    return img


def skin(img, inv=False):
    cy, cr, cb = cv2.split(img)
    th_y = thresh(cy, [54, 163], inv, 'Y')
    th_cr = thresh(cr, [131, 167], inv, 'Cr')
    th_cb = thresh(cb, [110, 135], inv, 'Cb')

    return cv2.merge((th_y, th_cb, th_cr))


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
fgbg = cv2.createBackgroundSubtractorMOG2()
lR = 1
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    frame = cv2.flip(frame, 1)

    cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    cv2.imshow("camera", frame)
    fgmask = fgbg.apply(frame, learningRate=lR)
    lR -= (lR / 2) if lR > 0.25 else lR

    skins = skin(frame)
    skins = cv2.morphologyEx(skins, cv2.MORPH_CLOSE, (5, 5))
    skins_nobg = cv2.bitwise_and(skins, skins, mask=fgmask)
    cv2.imshow("colored", skins_nobg)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    dup_frame = skins_nobg[:][:][:]
    for (x, y, w, h) in faces:
        dup_frame[y:y+h, x:x+w] = np.zeros((h, w, 3))

    dup_frame = cv2.cvtColor(dup_frame, cv2.COLOR_YCR_CB2RGB)
    dup_frame = cv2.cvtColor(dup_frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(dup_frame, 0, 255, L2gradient=True)
    alpha = 0.5
    beta = (1.0 - alpha)
    edged = cv2.addWeighted(edges, alpha, dup_frame, beta, 0.0)
    # cv2.imshow("edged", edged)

    edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, (5, 5))
    blur = cv2.GaussianBlur(edged, (5, 5), 0)
    _, edged = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, (5, 5))
    cv2.imshow("edged", edged)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
    # edged = cv2.cvtColor(edged, cv2.COLOR_RGB2YCR_CB)
    frame_contour = cv2.drawContours(edged, contours, -1, (0, 255, 0), 2)

    max_contour = max(contours, key=cv2.contourArea)
    # cv2.imshow("Xc", frame_contour)
    # frame_contour_clr = cv2.cvtColor(frame_contour, cv2.COLOR_GRAY2RGB)
    # for contour in contours:
    #     (x, y), radius = cv2.minEnclosingCircle(contour)
    #     center = (int(x), int(y))
    #     radius = int(radius)
    #     cv2.circle(frame_contour_clr, center, radius, (0, 255, 0), 2)
    # cv2.imshow("X", frame_contour_clr)
    #
    frame_contour = cv2.cvtColor(frame_contour, cv2.COLOR_GRAY2RGB)
    epsilon = 0.008 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    cv2.drawContours(frame_contour, [approx], -1, (0, 255, 0), 4)
    cv2.imshow("Output", frame_contour)
    c = cv2.waitKey(1)
    if c == 27:
        break
    elif c == ord('r'):
        lR = 1
cap.release()
