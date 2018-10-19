import numpy as np
import cv2
from imutils.video import WebcamVideoStream
import win32api, win32con
import time
import math

def left_click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
 

 
def right_click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade1 = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml')
eye_cascade2 = cv2.CascadeClassifier('haarcascade_mcs_eyepair_small.xml')
eye_cascade3 = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
eye_cascade4 = cv2.CascadeClassifier('haarcascade_eye.xml')

window_size = 5
windowX = []
windowY = []

cap = WebcamVideoStream(src=0).start()


# Release everything if job is finished
mx = 0
my = 0
nx=0
ny=0
k1 = 0
k2 = 0
count = 0
c=0

upper_mx = 0
lower_mx = 0

upper_my = 0
lower_my = 0

start_flag = True

while True:
    
    mx1=mx
    my1=my
    
    img = cap.read()
    img = cv2.flip(img, 1)

    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade1.detectMultiScale(gray)
    if len(eyes) == 0:
        eyes = eye_cascade2.detectMultiScale(gray)

    area = 0
    x = 0
    y = 0
    w = 0
    h = 0
        
    for (ex,ey,ew,eh) in eyes:
        if ew*eh > area:
            x = ex
            y = ey
            h = eh
            w = ew
            area = w*h

    if area > 0:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img,(int(x+w/2-5),int(y+h/2-5)),(int(x+w/2+5),int(y+h/2+5)),(255,0,0),2)
        mx = x + w/2
        my = y + h/2
        roi_gray = gray[y-20:y+h+20, x-20:x+w+20]
        roi_color = img[y-20:y+h+20, x-20:x+w+20]

        '''
        eyesLR = eye_cascade3.detectMultiScale(roi_gray)
        if len(eyesLR) == 0:
            eyesLR = eye_cascade4.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyesLR:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        '''
        
        if start_flag == True:
            upper_mx = mx + 35
            lower_mx = mx - 20
            upper_my = my + 35
            lower_my = my - 20
            start_flag = False
            
            
        mx = np.interp(mx,[lower_mx,upper_mx],[0,1366])
        my = np.interp(my,[lower_my,upper_my],[0,768])
        
        windowX.append(mx)
        windowY.append(my)

        if len(windowX) > window_size:
            del windowX[0]
        if len(windowY) > window_size:
            del windowY[0]
        
        
        
        mx = int(np.mean(windowX))
        my = int(np.mean(windowY))
        
        d=math.sqrt(math.pow(mx-mx1,2)+math.pow(my-my1,2))
        if d>15:
            nx=mx1+int(d/48)
            ny=my1+int(d/48)
            win32api.SetCursorPos((nx,ny))
        #time.sleep(0.02)
        print(mx1)
        print(my1)

    faces1 = face_cascade.detectMultiScale(gray, 1.3, 5)        
    for (xi, yi, wi, hi) in faces1:
        k2 = 1
        img = cv2.rectangle(img, (xi, yi), (xi + wi, yi + hi), (255, 0, 0), 2)
            
        #cir = cv2.circle(img, (int(x + w / 2),int( y + h / 2)), 1, (0, 255, 255), 2)
            
        roi_grayi = gray[yi:yi + hi, xi:xi + wi]
        roi_colori = img[yi:yi + hi, xi:xi + wi]
        eyesi = eye_cascade4.detectMultiScale(roi_grayi, 1.3, 5)
        for (exi, eyi, ewi, ehi) in eyesi:
            k2 = 0
            cv2.rectangle(roi_colori, (exi, eyi), (exi + ewi, eyi + ehi), (255, 255, 0), 2)        

        
        if k1==1 and k2==1:
            count=count+1
            c=c+1
        elif k2==0:
            k1=0
            count=0
            c=0
        elif k1==0 and k2==1:
            k1=1
            count=1
            c=1
        if k2 == 1:
            print("Blink")
            cv2.putText(img, "**************************BLINK!************************", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
            cv2.putText(img, "**************************BLINK!*************************", (10,460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255,0), 2)
        if count==2:
            left_click(nx,ny)
            count=0
        
        
        if c>=7:    
            print("drowsy")
            cv2.putText(img, "*********************DROWSY!************************", (10, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            from playsound import playsound
            playsound('beep-01a.wav')    
            c=0
        '''if c ==4:
            right_click(nx,ny)
            c=0'''
            
    time.sleep(0.1)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('img',img)
    print ("mx: " + str(mx) + " ! my: " + str(my))
    if cv2.waitKey(1) == ord('q'):
        break
    


cap.stop()
cv2.destroyAllWindows()