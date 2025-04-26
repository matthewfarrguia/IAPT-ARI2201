from ultralytics import YOLO #load yolov8 model
import cv2
model = YOLO('yolov8m.pt')


#load video
video = 'vid.mp4'
cap = cv2.VideoCapture(video)


#read frames from video
ret = True
while ret:
    ret, frame = cap.read()

    if ret:

        #apply object detection


        #track objects
        results = model.track(frame,persist=True)

        #plot results
        plot = results[0].plot()

        #visualize
        cv2.imshow('frame',plot)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break