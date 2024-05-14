from ultralytics import YOLO
import cv2
import math 
from sort import *

cap = cv2.VideoCapture("cars.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# write video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter("cars_count.mp4",fourcc, 20, (width, height))

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

vehicles = ["car", "motorbike", "bus","truck"]
mask = cv2.imread("mask.png")

tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.6)

incoming_count = []
outgoing_count = []

while cap.isOpened():
    ret, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    if ret == True:
        results = model(imgRegion, stream=True)

        detections = np.empty((0,5))

        # crossing boundary
        crossing_line = 400
        upper_boundary = 380
        lower_boundary = 420
        color = (0, 200, 0)
        cv2.line(img, (0, crossing_line), (int(width), crossing_line), color, 4)
        cv2.line(img, (0,upper_boundary), (int(width), upper_boundary), color, 1)
        cv2.line(img, (0,lower_boundary), (int(width), lower_boundary), color, 1)


        # frame divider
        x = 680
        y = 720
        # cv2.line(img, (680, 0), (680, 720), (0,200,100), thickness)

        
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding boxes
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                # print("x1,y1,x2,y2", x1,y1,x2,y2)


                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                # print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                # print("Class name -->", classNames[cls])


                # detect only required vehicles
                currentClass = classNames[cls]
                if currentClass in vehicles and confidence >= 0.4:
                # put box in cam
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                    currArray = np.array([x1,y1,x2,y2, confidence])
                    detections = np.vstack((detections, currArray))

        trackerResults = tracker.update(detections)
        print(trackerResults)

        for result in trackerResults:
            x1, y1, x2, y2, id = result 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            w, h = x2 - x1 , y2 - y1
            cx,cy = x1+w//2, y1+h//2

            # box details
            org = [x1, y1]
            
            if cx < x:
                cv2.rectangle(img, (x1, y1), (x2, y2), (70, 194, 203), 2)
            if cx > x:
                cv2.rectangle(img, (x1, y1), (x2, y2), (3, 3, 255), 2)

            # cv2.putText(img, f"{confidence}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 255), 2, 2)

            print(result)

            # object centre
            cv2.circle(img, (cx,cy), 3, (255, 0, 255), cv2.FILLED)

            if cx < x and upper_boundary <= cy <= lower_boundary and outgoing_count.count(id) == 0:
                outgoing_count.append(id)
            
            if cx > x and upper_boundary <= cy <= lower_boundary and incoming_count.count(id) == 0:
                incoming_count.append(id)
            

        cv2.putText(img, f"Outgoing Count: {len(outgoing_count)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 10, 10), 2, 2)
        cv2.putText(img, f"Incoming Count: {len(incoming_count)}", (900, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 10, 10), 2, 2)

        cv2.imshow("Image:", img)
        # cv2.imshow("ImageRegion:", imgRegion)
        out.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()