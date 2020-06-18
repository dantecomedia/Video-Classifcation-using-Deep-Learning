import numpy as np 
from utils import Videos
import cv2 
import keras
import model

reader = Videos(target_size=(128, 128), 
                to_gray=True, 
                max_frames=29, 
                extract_frames='middle', 
                normalize_pixels=(0, 1))

cap=cv2.VideoCapture(0)
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))

print(frame_width, frame_height)

while True:
	ret,frame=cap.read()
	if ret == True:
		cv2.imshow("frame", frame)
		r=reader.read_videos(frame)
		print(r)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	else:
		break
cap.release()
cv2.destroyAllWindows()
