import cv2 as cv
import numpy as np
import streamlit as st

st.title("Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    src = cv.imdecode(file_bytes, 1)
    src = cv.resize(src, None, fx=0.3, fy=0.3)

    imgBlur = cv.medianBlur(src, 11)

    Gray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)
    ret3, th3 = cv.threshold(Gray, 90, 255, cv.THRESH_BINARY)

    hsv = cv.cvtColor(imgBlur, cv.COLOR_BGR2HSV)
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([60, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))

    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    opening_yellow = cv.morphologyEx(mask_yellow, cv.MORPH_OPEN, kernel1, iterations=1)
    closed_yellow = cv.morphologyEx(opening_yellow, cv.MORPH_CLOSE, kernel1, iterations=1)

    mask_green = cv.inRange(hsv, lower_green, upper_green)
    opening_green = cv.morphologyEx(mask_green, cv.MORPH_OPEN, kernel1, iterations=1)
    closed_green = cv.morphologyEx(opening_green, cv.MORPH_CLOSE, kernel1, iterations=1)

    contours, _ = cv.findContours(closed_green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours1, _ = cv.findContours(closed_yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    s1 = s2 = 0
    for contour in contours1:
        pr = cv.arcLength(contour, True)
        M = cv.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if pr < 250:
            s1 += 1
            cv.drawContours(src, contour, -1, (0, 255, 0), 3)
            cv.putText(src, "Y(s)", (cX - 25, cY - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            s2 += 1
            cv.drawContours(src, contour, -1, (255, 0, 0), 3)
            cv.putText(src, "Y(L)", (cX - 25, cY - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    n1 = n2 = 0
    for contour in contours:
        pr = cv.arcLength(contour, True)
        M = cv.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if pr < 250:
            n1 += 1
            cv.drawContours(src, contour, -1, (0, 255, 0), 3)
            cv.putText(src, "G(s)", (cX - 25, cY - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            n2 += 1
            cv.drawContours(src, contour, -1, (255, 0, 0), 3)
            cv.putText(src, "G(L)", (cX - 25, cY - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    st.image(src, channels="BGR", caption='Object detection')

    st.write("จำนวนมะนาวสีเขียวทั้งหมด =", len(contours))
    st.write("สีเขียวขนาดเล็ก =", n1)
    st.write("สีเขียวขนาดใหญ่ =", n2)
    st.write("จำนวนมะนาวสีเเหลืองทั้งหมด =", len(contours1))
    st.write("สีเเหลืองขนาดเล็ก =", s1)
    st.write("สีเเหลืองขนาดใหญ่ =", s2)
