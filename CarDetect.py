# coding:utf8
import cv2
import sys

class Car():
    def __init__(self,x,y,w,h) -> None:
        self.centerX = int(x+(w/2))
        self.centerY = int(y+(h/2))
        self.Updated = False
        self.Counted = False

class existedCars():
    '''
    Create the information map for existed car on current frame,
    you need to flush() after finish all Update in current frame.
    '''
    Cars = []
    def __init__(self):
        pass

    def flush(self):
        for item in self.Cars:
            if item.Updated:
                item.Updated = False
            else:
                self.Cars.remove(item)

    def Update(self,Car,L2diff):
        '''
        - L2diff : L2 distance of previous and current picture's car 
        '''
        Nearest = sys.maxsize
        rcd_idx = None
        for idx,item in enumerate(self.Cars):
            L2dist = ((item.centerX-Car.centerX)**2)+((item.centerY-Car.centerY)**2)
            if(L2dist<Nearest):
                Nearest = L2dist
                rcd_idx = idx
        if Nearest < L2diff:
            self.Cars[rcd_idx].centerX = Car.centerX
            self.Cars[rcd_idx].centerY = Car.centerY
            self.Cars[rcd_idx].Updated = True
        else:
            Car.Updated = True
            self.Cars.append(Car)

    def CarsInBoundedY(self,TopY,BotY):
        result = 0
        for item in filter(lambda Car:not(Car.Counted),self.Cars):
            if(item.centerY>TopY and item.centerY<BotY):
                item.Counted = True
                result += 1
        return result



def main():
    cap=cv2.VideoCapture(r'S_8418073834736.mp4') #Sample Video's path
    fgbg =cv2.createBackgroundSubtractorMOG2() #Apply BS method from Opencv
    ret, frame = cap.read()
    high, width = frame.shape[:2]
    detectY = int(high/2) + 15 #Set the position of Y axis of detect line
    carNum = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    linecounter = 2
    CarRecorder = existedCars()

    while(1):
        ret,frame=cap.read()
        if ret == 0:
            break
        fgmask=fgbg.apply(frame)
        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
        fgmask = cv2.threshold(fgmask.copy(), 150, 255, cv2.THRESH_BINARY)[1]
        fgmask = cv2.dilate(fgmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=4)
        fgmask = cv2.erode(fgmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

        contours, hier = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 2500: #filter small area to focus on the target
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
                carCenterY = int(y+(h/2))
                if (carCenterY > detectY - 17 and carCenterY < detectY + 17 and linecounter >= 2):
                    cv2.line(frame, (0, detectY), (width, detectY), (255, 0, 0), 3)
                    linecounter = 0
                
                CarRecorder.Update(Car(x, y, w, h),800) #Update information map
        carNum += CarRecorder.CarsInBoundedY(detectY - 17, detectY + 17) #Get car number in the target region
        CarRecorder.flush()
        linecounter += 1
        if linecounter >= 2:
            cv2.line(frame, (0, detectY), (width, detectY), (0, 0, 255), 3)
        cv2.putText(frame, "Car Numer: %d" % carNum, (0, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("BackgroundSubtractorMOG2", fgmask)
        cv2.imshow("Car detection result", frame)

        if cv2.waitKey(50) & 0xff == 0x1B:
            break

    print("Total counted car number:", carNum)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
