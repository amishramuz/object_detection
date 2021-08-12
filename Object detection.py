import numpy as np
import cv2

thres = 0.55 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress 
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,280) #width 
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,120) #height 
# cap.set(cv2.CAP_PROP_BRIGHTNESS,150) #brightness 

classNames = []
with open('Labels.txt','r') as f:
    classNames=f.read().splitlines()

font = cv2.FONT_HERSHEY_PLAIN
#font = cv2.FONT_HERSHEY_COMPLEX
Colors = np.random.uniform(0, 255, 3)

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success,frame = cap.read()
    
    classIds, confs, bbox = net.detect(frame,thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    if len(classIds) != 0:
        for i in indices:
            i = i[0]
            box = bbox[i]
            confidence = str(round(confs[i],2))
            color = (255,0,0)
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, thickness=2)
            cv2.putText(frame, classNames[classIds[i][0]-1]+" "+confidence,(x+10,y+20),
                        font,1,color,2)
#             cv2.putText(img,str(round(confidence,2)),(box[0]+100,box[1]+30),


    cv2.imshow('object detection',frame )
    if cv2.waitKey(20)  & 0xFF ==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()