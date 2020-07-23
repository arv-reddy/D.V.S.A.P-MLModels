import sys
import json
import base64
import time
import cv2
import numpy as np
from kafka import KafkaProducer
from multiprocessing import Process
import pymongo
import imutils

# myclient = pymongo.MongoClient("mongodb://localhost:27017/")

# mydb = myclient["mydatabase"]

# table = mydb["camera_urls"]

# camera_urls = [str(d['link']) for d in table.find({},{"_id":0})]

# print(camera_urls)

# rtsp links statically initialized
camera_urls = ["pedestrians.mp4","example_01.mp4"]

# topic to write to
topic="video"

def np_to_json(obj):
	return {"frame":base64.b64encode(obj.tostring()).decode("utf-8")}

class StreamVideo(Process):
	
	def __init__(self,video_path,cam_num):
		"""
			  Video Streaming Producer Process Class. Publishes frames from a video source to a topic.
			  :param video_path: video url(rtsp)
			  :param cam_num: used in key to determine partition
		"""
		super(StreamVideo,self).__init__()
		self.video_path=video_path
		self.cam_num=cam_num

	def run(self):
		""" Publish video frames as bytes """
		
		# Producer Object
		producer = KafkaProducer(bootstrap_servers = 'localhost:9092',
														  value_serializer=lambda value: json.dumps(value).encode())

		camera = cv2.VideoCapture(self.video_path)

		totalFrames = 0
		# Read frame-by-frame and publish
		while True:
			if totalFrames%100 == 0:
				success,frame = camera.read()
				# msg = self.transform(frame)
				string = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()
				dict = {'frame': string}
				producer.send(topic,partition=self.cam_num,value=dict)
				time.sleep(0.2)
			totalFrames += 1

		camera.release()

	def transform(self,frame):
		# frame = imutils.resize(frame, width=400)
		frame_dict = np_to_json(frame.astype(np.uint8))
		msg = {"camera": self.cam_num}
		msg.update(frame_dict)
		return msg
		# return frame_dict

# Init StreamVideo processes, these publish frames from respective camera to the same topic
PRODUCERS = [StreamVideo(url,index) for index,url in enumerate(camera_urls)]

# Start Publishing frames from cameras to the frame topic
for p in PRODUCERS:
	p.start()
	
# wait for producer processes to end
for p in PRODUCERS:
	p.join()