#เฉพาะสีเขียว  ตัวเลขเส้นรอบรูป+นับแยกขนาด using findContours + arcLength + moments
import sys
import cv2 as cv
import numpy as np

# Loads an image
src = cv.imread('C:/Users/KeNg/Downloads/New folder (4)/testcolor.JPG')
src =cv.resize(src , None , fx=0.3 , fy=0.3)
#row,col,ch = src.shape
imgBlur = cv.medianBlur(src, 11)
#print(row,col)
####### แปลงเป็นรูปสีเทา (gray scale) #########
Gray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)

###### แปลงเป็นภาพขาวดำ #####
ret3,th3 = cv.threshold(Gray,90,255,cv.THRESH_BINARY)

####### แปลงเป็นสี HSV #########
hsv = cv.cvtColor(imgBlur, cv.COLOR_BGR2HSV)
lower_green = np.array([30, 50, 50])   
upper_green = np.array([60, 255, 255])   
lower_yellow = np.array([20,100,100]) 
upper_yellow = np.array([30,255,255])  #[27,255,255]

###### หาพิกเซลในรูปที่มี ช่วงสี hsv ที่กำหนด ######
kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11))
##### สีเหลือง ####
mask_yellow = cv.inRange(hsv, lower_yellow , upper_yellow)
opening_yellow = cv.morphologyEx(mask_yellow, cv.MORPH_OPEN, kernel1, iterations = 1)
closed_yellow = cv.morphologyEx(opening_yellow, cv.MORPH_CLOSE, kernel1, iterations = 1)

##### สีเขียว ####
mask_green = cv.inRange(hsv, lower_green , upper_green)
opening_green = cv.morphologyEx(mask_green, cv.MORPH_OPEN, kernel1, iterations = 1)
closed_green = cv.morphologyEx(opening_green, cv.MORPH_CLOSE, kernel1, iterations = 1)

contours,h = cv.findContours(closed_green, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_NONE)
contours1,s = cv.findContours(closed_yellow, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_NONE)
#print(contours)
s1=s2=0
for contour in contours1:
        pr= cv.arcLength(contour,True)
        M = cv.moments(contour) 
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if pr < 250 :
            s1=s1+1
            cv.drawContours(src, contour, -1, (0, 255, 0), 3)
            cv.putText(src, "Y(s)", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else :
            s2=s2+1
            cv.drawContours(src, contour, -1, (255, 0, 0), 3)
            cv.putText(src, "Y(L)", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

n1=n2=0
for contour in contours:
        pr= cv.arcLength(contour,True)
        M = cv.moments(contour) 
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if pr < 250 :
            n1=n1+1
            cv.drawContours(src, contour, -1, (0, 255, 0), 3)
            cv.putText(src, "G(s)", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else :
            n2=n2+1
            cv.drawContours(src, contour, -1, (255, 0, 0), 3)
            cv.putText(src, "G(L)", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
cv.imshow("Object detection", src)
print("จำนวนมะนาวสีเขียวทั้งหมด = " + str(len(contours))) 
print("สีเขียวขนาดเล็ก = " , n1)
print("สีเขียวขนาดใหญ่ = " , n2)
print("จำนวนมะนาวสีเเหลืองทั้งหมด = " + str(len(contours1))) 
print("สีเเหลืองขนาดเล็ก = " , s1)
print("สีเเหลืองขนาดใหญ่ = " , s2)
cv.waitKey(0)
cv.destroyAllWindows()