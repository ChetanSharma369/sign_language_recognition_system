import cv2
import mediapipe as mp
import handtest as hld
import numpy as np
import time
import os


class handFiles:
    def __init__(self, folderpath):
        self.folderpath = folderpath
        self.mylist = os.listdir(folderpath)
        self.overlaylist = []
        if len(self.mylist) != 0:
            self.loadFiles()
    
    def loadFiles(self):
        """Load and resize all images from folder"""
        self.overlaylist = []
        for img in self.mylist:
            image = cv2.imread(f'{self.folderpath}/{img}')
            image = cv2.resize(image, (200, 200), cv2.INTER_LINEAR)
            self.overlaylist.append(image)
        return self.overlaylist


def draw_grid(img, grid_space=50, color=(255, 255, 255), thickness=1):
    """Draw grid overlay on image"""
    h, w = img.shape[:2]
    
    # Draw vertical lines
    for x in range(0, w, grid_space):
        cv2.line(img, (x, 0), (x, h), color, thickness)
    
    # Draw horizontal lines
    for y in range(0, h, grid_space):
        cv2.line(img, (0, y), (w, y), color, thickness)


def draw_ui_labels(img, mode=" "):
    """Draw UI text labels on image"""
    cv2.putText(img, 'BRUSH', (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, 'ERASE', (1100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if mode != " ":
        cv2.putText(img, f'{mode} mode', (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def handle_selection_mode(img, imgCanvas, x1, y1, fingers):
    """Handle mode selection (brush or eraser)"""
    flag = " "
    if fingers[1] and fingers[2]:
        cv2.circle(img, (x1, y1), 15, (0, 225, 0), cv2.FILLED)
        if y1 < 125:
            if 150 < x1 < 250:  # brush selected
                flag = "brush"
            elif 900 < x1 < 1100:  # eraser selected
                flag = "eraser"
    return flag


def handle_brush_mode(img, imgCanvas, x1, y1, xp, yp, fingers):
    """Handle drawing with brush"""
    if fingers[1] and not fingers[2]:
        if xp == 0 and yp == 0:
            xp, yp = x1, y1
        cv2.rectangle(img, (x1 - 15, y1 - 15), (x1 + 15, y1 + 15), (0, 255, 0), cv2.FILLED)
        cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 255), 5)
        xp, yp = x1, y1
    return xp, yp


def handle_eraser_mode(img, imgCanvas, x1, y1, xp, yp, fingers):
    """Handle erasing with eraser"""
    if fingers[1] and not fingers[2]:
        if xp == 0 and yp == 0:
            xp, yp = x1, y1
        cv2.rectangle(img, (x1 - 15, y1 - 15), (x1 + 15, y1 + 15), (0, 0, 0), cv2.FILLED)
        cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), 50)
        xp, yp = x1, y1
    return xp, yp


if __name__ == '__main__':
    # Initialize variables
    flag = " "
    folderpath = 'headers'
    
    # Setup hand detection and video capture
    handImage = handFiles(folderpath)
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    detector = hld.handDetector()
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    
    xp, yp = 0, 0
    
    # Main loop
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        if not ret:
            continue
        
        # Find hand landmarks
        img = detector.findHands(frame)
        draw_grid(img, grid_space=100, color=(255, 0, 0), thickness=1)
        draw_grid(imgCanvas, grid_space=100, color=(255, 0, 0), thickness=1)
        
        data = detector.handLandmarks(frame)
        fingers = detector.FingerCheck(data)
        
        # Draw UI on both images
        draw_ui_labels(img, flag)
        draw_ui_labels(imgCanvas, flag)
        
        # Process hand data
        if len(data) != 0:
            x1, y1 = data[8][1:]  # Index finger tip
            x2, y2 = data[12][1:]  # Middle finger tip
            
            # Selection mode
            new_flag = handle_selection_mode(img, imgCanvas, x1, y1, fingers)
            if new_flag != " ":
                flag = new_flag
                xp, yp = 0, 0  # Reset drawing position
            
            # Drawing mode
            if flag == "brush":
                xp, yp = handle_brush_mode(img, imgCanvas, x1, y1, xp, yp, fingers)
            
            # Erasing mode
            elif flag == "eraser":
                xp, yp = handle_eraser_mode(img, imgCanvas, x1, y1, xp, yp, fingers)
        
        # Display images
        cv2.imshow('live data', img)
        cv2.imshow('canvas', imgCanvas)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
        
            

        



