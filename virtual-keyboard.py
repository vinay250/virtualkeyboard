import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from time import sleep 
import numpy as np
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0)

# Set Video Dimensions
cap.set(3,1280)
cap.set(4,720)

# Hand tracker
detector = HandDetector(staticMode=False,maxHands=1,detectionCon=0.8)

def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),                         20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]),(255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 40, y + 60),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out
    
    

# Defining a class
class Button():
    def __init__(self,pos,text,size=(85,85)):
        self.pos = pos
        self.text = text
        self.size = size
        
buttonList = []

keys=[['Q','W','E','R','T','Y','U','I','O','P'],
      ['A','S','D','F','G','H','J','K','L',';'],
      ['Z','X','C','V','B','N','M',',','.','/']]

keyboard = Controller()

for i in range(len(keys)):
            for j,key in enumerate(keys[i]):
                buttonList.append(Button((100*j+50,100*i+50),key))
                
while True:
    success,img = cap.read()

    hands , img = detector.findHands(img)

    if hands:
        hand = hands[0] 
        lmList = hand["lmList"]  
        bbox = hand["bbox"]  
        center = hand['center']  
        handType = hand["type"]

        img = drawAll(img,buttonList)

        for button in buttonList:
            x,y = button.pos
            w,h = button.size

            if x<lmList[8][0]<x+w and y<lmList[8][1]<y+h:
                cv2.rectangle(img,button.pos,(x+w,y+h),(175,0,175),cv2.FILLED)
                cv2.putText(img,button.text,(x+20,y+65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)

                # Calculate distance between specific landmarks on the first hand and draw it on the image
                length, info, img = detector.findDistance(lmList[8][0:2],lmList[12][0:2],img,color=(255, 0, 255),draw=False)
                
                if length<45:
                    keyboard.press(button.text)
                    cv2.rectangle(img,button.pos,(x+w,y+h),(0,255,0),cv2.FILLED)
                    cv2.putText(img,button.text,(x+20,y+65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
                    sleep(0.2)
  
                

    cv2.imshow("Img",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break