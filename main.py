import cv2
import numpy as np
import matplotlib.pyplot as plt
def make_coordinates(image,line_parameters):
   slope,intercept=line_parameters
   y1=image.shape[0]
   y2=int(y1*(3/5))
   x1=int((y1-intercept)/slope)
   x2=int((y2-intercept)/slope)
   return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
   left_fit=[]
   right_fit=[]
   for line in lines:
      x1,y1,x2,y2=line.reshape(4)
      parameters=np.polyfit((x1,x2),(y1,y2),1)
      slope=parameters[0]
      intercept=parameters[1]
      if slope<0:
         left_fit.append((slope,intercept))
      else:
         right_fit.append((slope,intercept))
   left_fit_average=np.average(left_fit,axis=0)
   right_fit_average=np.average(right_fit,axis=0)
   left_line=make_coordinates(image,left_fit_average)
   right_line=make_coordinates(image,right_fit_average)
   return np.array([left_line,right_line])

def canny(image):
   gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
   #reduce noise - using gaussian blur
   blur=cv2.GaussianBlur(gray,(5,5),0)
   #doing canny to identify the lines
   canny=cv2.Canny(blur,50,150)
   return canny

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
      for line in lines:
         #convert 2d line to 1d line
         x1,y1,x2,y2=line.reshape(4)
         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image
   
def region_of_interest(image):
   height=image.shape[0]
   polygons=np.array([[(200,height),(1100,height),(550,250)]
      ])
   #a black image is create having the same height and weight as the original image that is the use of zeoes_like
   mask=np.zeros_like(image)
   cv2.fillPoly(mask,polygons,255)
   #used for isolation of region of interest
   masked_image=cv2.bitwise_and(image,mask)
   return masked_image


#image=cv2.imread('test_image.jpg')
#lane_image=np.copy(image)
#the image is converted to greyscale
#canny = canny(lane_image)
#plt.imshow(canny)
#cropped_image=region_of_interest(canny)
#lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
#averaged_lines = average_slope_intercept(lane_image,lines)
#line_image=display_lines(lane_image,averaged_lines)
#detected lines into the orginal image
#combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
#cv2.imshow("lane detection by anjana",combo_image)
#cv2.waitKey(0)
#plt.show()
cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
   _,frame=cap.read()
   canny_image = canny(frame)
   #plt.imshow(canny)
   cropped_image=region_of_interest(canny_image)
   lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
   averaged_lines = average_slope_intercept(frame,lines)
   line_image=display_lines(frame,averaged_lines)
   #detected lines into the orginal image
   combo_image=cv2.addWeighted(frame,0.8,line_image,1,1)
   cv2.imshow("lane detection by anjana",combo_image)
   cv2.waitKey(1)
