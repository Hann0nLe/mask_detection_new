# import cv2,os,urllib.request
# import numpy as np
# from django.conf import settings
# face_detection_videocam = cv2.CascadeClassifier(os.path.join(
# 			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
# face_detection_webcam = cv2.CascadeClassifier(os.path.join(
# 			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
#
# class VideoCamera(object):
# 	def __init__(self):
# 		self.video = cv2.VideoCapture(1)
#
# 	def __del__(self):
# 		self.video.release()
#
# 	def get_frame(self):
# 		success, image = self.video.read()
# 		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
# 		# so we must encode it into JPEG in order to correctly display the
# 		# video stream.
#
# 		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 		faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# 		for (x, y, w, h) in faces_detected:
# 			cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
# 		frame_flip = cv2.flip(image,1)
# 		ret, jpeg = cv2.imencode('.jpg', frame_flip)
# 		return jpeg.tobytes()
#
#
# class IPWebCam(object):
# 	def __init__(self):
# 		self.url = "http://192.168.0.100:8080/shot.jpg"
#
#
# 	def __del__(self):
# 		cv2.destroyAllWindows()
#
# 	def get_frame(self):
# 		imgResp = urllib.request.urlopen(self.url)
# 		imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
# 		img= cv2.imdecode(imgNp,-1)
# 		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
# 		# so we must encode it into JPEG in order to correctly display the
# 		# video stream
# 		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 		faces_detected = face_detection_webcam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# 		for (x, y, w, h) in faces_detected:
# 			cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
# 		resize = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR)
# 		frame_flip = cv2.flip(resize,1)
# 		ret, jpeg = cv2.imencode('.jpg', frame_flip)
# 		return jpeg.tobytes()

from keras.models import load_model
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
from pygame import mixer
import time
model = load_model('model-017.model')
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
mixer.init()
sound = mixer.Sound('alarm.wav')
face_detection_videocam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
face_detection_webcam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(1)


	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		for (x, y, w, h) in faces_detected:
			face_img=gray[y:y+w,x:x+w]
			resized=cv2.resize(face_img,(100,100))
			normalized=resized/255.0
			reshaped=np.reshape(normalized,(1,100,100,1))
			result=model.predict(reshaped)
			label=np.argmax(result,axis=1)[0]
	#         print(label)
			cv2.rectangle(image,(x,y),(x+w,y+h),color_dict[label],2)
			cv2.rectangle(image,(x,y-40),(x+w,y),color_dict[label],-1)
			cv2.putText(image, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
			if label==1:
				sound.play()
			else:
				sound.stop()
		frame_flip = cv2.flip(image,1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()


class IPWebCam(object):
	def __init__(self):
		self.url = "http://192.168.0.100:8080/shot.jpg"


	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self):
		imgResp = urllib.request.urlopen(self.url)
		imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
		img= cv2.imdecode(imgNp,-1)
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces_detected = face_detection_webcam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		for (x, y, w, h) in faces_detected:
			cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
		resize = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR)
		frame_flip = cv2.flip(resize,1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()
