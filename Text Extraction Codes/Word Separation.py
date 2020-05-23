
# coding: utf-8

# In[51]:


import cv2
import numpy as np
from cv2 import line,resize, imwrite, imshow, boundingRect, countNonZero, cvtColor, drawContours, findContours, getStructuringElement, imread, morphologyEx, pyrDown, rectangle, threshold


# In[61]:


img = imread(r"C:\Users\u293217\Downloads\2.png") 
img = cv2.transpose(img)  
img = cv2.flip(img, 1)


# In[62]:


row1, col1= img.shape[:2]


# In[63]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
imwrite(r"C:\Users\u293217\Downloads\\grayscale.jpg", gray)
gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,1201,1)
imwrite(r"C:\Users\u293217\Downloads\\threshold.jpg", gray)


# In[64]:


kernel = np.ones((5,5),np.uint8)
gray = cv2.erode(gray,kernel,iterations = 3)
imwrite(r"C:\Users\u293217\Downloads\\erode.jpg", gray)
gray = cv2.dilate(gray,kernel,iterations = 3)
imwrite(r"C:\Users\u293217\Downloads\\dilate.jpg", gray)


# In[65]:


row, col= img.shape[:2]
bottom= img[row-2:row, 0:col]
mean= cv2.mean(bottom)[0]
bordersize=50
border1=cv2.copyMakeBorder(gray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
border2=cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )


# In[66]:


im2, contours, hierarchy = cv2.findContours(border1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in range(1,len(contours)):    
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if area > 200:
        roi2 = border2[y:y+h, x:x+w]
        roi2 = cv2.transpose(roi2)
        roi2 = cv2.flip(roi2,0)
        imwrite(r"C:\Users\u293217\Downloads\\"+str(i)+".jpg", roi2)

