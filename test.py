import os
import cv2
#from extract_face import extract_face
from mtcnn.mtcnn import MTCNN 
from datetime import datetime
import numpy as np
import uuid 
import csv
import time 
from threading import Thread
from queue import Queue
now=datetime.now()
detector = MTCNN()

cap=cv2.VideoCapture("Rush hours in india-Bangladesh-China and Japan.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while True :
	start_time=time.time()
	ret,frame = cap.read()
	#frame = np.dstack([frame, frame, frame])
	#frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	result = detector.detect_faces(frame)
	
	count=len(result)
	font = cv2.FONT_HERSHEY_SIMPLEX 
	 
	frame = cv2.putText(frame,"Face Count:"+str(count),(10,50), font,  1, (255,255,255), 2, cv2.LINE_AA) 

	if result!=[]:
		for person in result :
			bounding_box=person['box']
			keypoints=person['keypoints']
			x1,y1, width, height= person['box']
			x1,y1= abs(x1),abs(y1)
			x2,y2=x1+width,y1+height
			face=frame[y1:y2,x1:x2]
			current_time = now.strftime("%H:%M:%S")
			filename= str(uuid.uuid4())+".jpg"
			#writer.writerow({'File_Name':filename,'Time':current_time})
			path ="/media/rosa-mysitca/PYTHON/Security/Face-Recognition-/Automatic Face Dataset/Face_dataset/"
			filename=path+filename
			cv2.imwrite(filename,face)
			frame=cv2.rectangle(frame,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
			frame = cv2.putText(frame," FPS: {:.2f}".format(1.0/(time.time()-start_time)),(10,30), font,  1, (255,255,255), 2, cv2.LINE_AA)
			out.write(frame)
#	cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
#	cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
#	cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
#	cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
#	cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)

	#display resulting frame

	#frame = cv2.putText(frame," FPS: {:.2f}".format(1.0/(time.time()-start_time)),(10,30), font,  1, (255,255,255), 2, cv2.LINE_AA)
	#cv2.imshow('frame',frame)


	#if cv2.waitKey(1) & 0xFF == ord('q') :
	#break 

	#cap.release()
	#cv2.destroyAllWindows()

