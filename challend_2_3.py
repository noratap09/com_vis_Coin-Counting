import numpy as np
import cv2
import glob

path = glob.glob("CoinCounting\*.jpg")
for cat in path:
    im = cv2.imread(cat)
    im =  cv2.resize(im,(500,500))

    im_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    im_mark = cv2.inRange(im_hsv,(93,30,30),(126,255,255))

    im_mark = cv2.medianBlur(im_mark,7)

    #blue
    dist_transform = cv2.distanceTransform(im_mark,cv2.DIST_L2,3)
    dist_transform = cv2.normalize(dist_transform,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

    ret , sure_fg = cv2.threshold(dist_transform,0.52*dist_transform.max(),255,0)

    sure_fg = cv2.medianBlur(sure_fg,5)

    contours,hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(im, contours, -1, (0, 255, 0), 2)

    for i in range(0,len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i]) #หากรอบสี่เหลียม
        temp = sure_fg[y:y+h,x:x+w]
        n = np.sum(temp)
        #cv2.putText(im,str(n),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
        if(n<50000):
            kernel = np.ones((5,5),np.uint8)
            temp = cv2.dilate(temp,kernel,iterations=3)

        sure_fg[y:y+h,x:x+w] = temp

    kernel = np.ones((5,5),np.uint8)
    sure_fg = cv2.erode(sure_fg,kernel)

    contours,hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, contours, -1, (0, 255, 0), 2)

    for i in range(0,len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i]) #หากรอบสี่เหลียม
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(im,"blue:"+str(i+1),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))

    #yellow
    im_mark = cv2.inRange(im_hsv,(20,100,100),(41,255,255))
    im_mark = cv2.medianBlur(im_mark,13)

    dist_transform2 = cv2.distanceTransform(im_mark,cv2.DIST_L2,3)
    dist_transform2 = cv2.normalize(dist_transform2,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

    ret , sure_fg_2 = cv2.threshold(dist_transform2,0.4*dist_transform2.max(),255,0)

    contours,hierarchy = cv2.findContours(sure_fg_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(im, contours, -1, (0, 255, 0), 2)

    for i in range(0,len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i]) #หากรอบสี่เหลียม
        temp = sure_fg_2[y:y+h,x:x+w]
        n = np.sum(temp)
        if(n>500000):
            kernel = np.ones((3,3),np.uint8)
            temp = cv2.erode(temp,kernel,iterations=3)

        sure_fg_2[y:y+h,x:x+w] = temp

    contours,hierarchy = cv2.findContours(sure_fg_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, contours, -1, (0, 255, 0), 2)
    for i in range(0,len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i]) #หากรอบสี่เหลียม
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(im,"yellow:"+str(i+1),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))



    cv2.imshow("im",im)
    cv2.imshow("dist_transform_yellow",dist_transform2)
    cv2.imshow("dist_transform_blue",dist_transform)
    #cv2.imshow("sure",sure_fg_2)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
