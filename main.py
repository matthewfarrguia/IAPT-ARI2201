from ultralytics import YOLO
import cv2


#load yolov8 model
model = YOLO('yolov8m.pt')


#load video
video = 'vid.mp4'
cap = cv2.VideoCapture(video)

index = 0   #index for frame counting initialized at 0

#read frames from video
ret = True  #represents whether frames are loaded (boolean)
while ret:  #while frames are being loaded

    ret, frame = cap.read()

    if ret: #extra safety measure

        index += 1
        if index % 5 != 0:  #only consider every 5th frame

            continue

        #track objects
        results = model.track(source = frame,conf = 0.0,
                              iou = 0.45, stream = False)

        res = results[0]

        #filtering objects
        cat = 15    #yolov8 class has class 15 for cat representation
        boxes = res.boxes.data.tolist() #convert to lists

        for box in boxes:   #loop for each detection

            x1,y1,x2,y2,conf,clas = box
            clas = int(clas)    #convert to integer if it is not already

            if clas == cat: #if detected cats, output green bounding box

                color = (0,255,0)
                label = f"{model.names[clas]} {conf:.2f}"

            else:

                if conf < 0.5:  #only objects with higher confidence score than 0.5

                    label = f"{model.names[clas]} {conf:.2f}"
                    continue

            #print rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 8. Draw a label
            label = f"{model.names[clas]} {conf:.2f}"
            cv2.putText(frame, label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        #visualize
        cv2.imshow('frame',label)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

#clean up
cv2.release()
cv2.destroyAllWindows()