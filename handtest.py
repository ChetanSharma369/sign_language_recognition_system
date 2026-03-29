import cv2
import mediapipe as mp
import time

class handDetector:
    def __init__(self, static_image_mode=False, maxHands=1, confidence=0.7, trackCon=0.5):
        self.static_image_mode = static_image_mode
        self.maxHands = maxHands
        self.confidence = confidence
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.confidence,
            min_tracking_confidence=self.trackCon
        )
        self.mp_draw = mp.solutions.drawing_utils



    def findHands(self, frame,draw=True):
        self.image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.result=self.hands.process(self.image)
        
        if self.result.multi_hand_landmarks:
            for handlm in self.result.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, handlm, self.mp_hands.HAND_CONNECTIONS)
        return frame 

    def handLandmarks(self,frame,draw=True):
        LmList=[]
        if self.result.multi_hand_landmarks:
            for handlm in self.result.multi_hand_landmarks:
                for id,lm in enumerate(handlm.landmark):
                    h,w,c=frame.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    LmList.append([id,cx,cy])
        return LmList

    def FingerCheck(self,landmarkList):
        fingers=[4,8,12,16,20]
        result=[]
        if len(landmarkList) != 0:
            
            for id in fingers:
                if id == 4:
                    if landmarkList[fingers[0]][1] < landmarkList[fingers[0]-1][1]:
                            result.append(1)
                    else:
                        result.append(0)
                elif landmarkList[id][2] < landmarkList[id-2][2]:
                    result.append(1)   
                else:
                    result.append(0)
        return result
        


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    hand = handDetector()
    while True:
        Ptime=time.time()
        ret, frame = cap.read()
        img = hand.findHands(frame)
        handlm=hand.handLandmarks(frame)
        print(hand.FingerCheck(handlm))
       # print(handlm)
        cv2.imshow("live hand", img)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(f'test_{int(Ptime)}.jpg',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break   

