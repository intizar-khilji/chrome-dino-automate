import cv2 as cv
import numpy as np
from PIL import ImageGrab
import time
from pynput.keyboard import Key, Controller

def region_of_interest(image, vertics):
    mask = np.zeros_like(image)
    match_mask_color = 255
    cv.fillPoly(mask, vertics, match_mask_color)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

keyboard = Controller()


speed_factor = 280

while True:
    image = ImageGrab.grab()
    frame = np.array(image)
    frame = cv.resize(frame, (1280,720))
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    region_of_interest_vertics1 = [(0,100), (100,100), (100,200), (0,200)]
    masked_image1 = region_of_interest(frame, np.array([region_of_interest_vertics1], np.int32))
    
    if (255,255,255) in masked_image1:
        lb = np.array([0, 0, 76])
        hb = np.array([170, 100, 100])
        mask = cv.inRange(hsv, lb, hb)
    else:
        lb = np.array([0, 0, 139])
        hb = np.array([0, 0, 184])
        mask = cv.inRange(hsv, lb, hb)
    
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones([5,5], np.uint8))                
    # region_of_interest_vertics = [(0,200), (500,200), (500,500), (0,500)]        
    region_of_interest_vertics = [(0,200), (600,200), (600,500), (0,500)]        
    masked_image = region_of_interest(mask, np.array([region_of_interest_vertics], np.int32))

    contours, _ = cv.findContours(masked_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    try:        
        if 3000 > cv.contourArea(contours[0]) > 2700:
            x1, y1, w1, h1 = cv.boundingRect(contours[0])
            cv.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0,255,0), 2)


        if cv.contourArea(contours[1]) > 700:
            x2, y2, w2, h2 = cv.boundingRect(contours[1])
            cv.rectangle(frame, (x2,y2), (x2+w2, y2+h2), (255,0,0))

            x1_center = (x1+w1, y1+h1//2)
            x2_center = (x2, y2+h2//2)
            cv.line(frame, x1_center, x2_center, (0,0,255), 2)
            distance = np.math.sqrt(((x1+w1)-(x2))**2 + ((y1+h1//2)-(y2+h2//2))**2)

            print(round(distance), speed_factor)
            if round(distance) < speed_factor and -6<(y2//2-y1//2) < 15:
                keyboard.press(Key.space) 
                # time.sleep(0.1)
                keyboard.release (Key.space)
            if round(distance) < 360   and (y2//2-y1//2) < -6:
                keyboard.press(Key.down)
                time.sleep(0.35) 
                keyboard.release (Key.down)
            speed_factor += 0.5
    except Exception as e: 
        pass
        # print(e)
    # cv.imshow('mask', mask)
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()
