from model1 import detect
from model2 import count
from people_counting.pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import FPS
from social_distance_detector.pyimagesearch import social_distancing_config as config
from flask import Flask, Response, render_template
from kafka import KafkaConsumer, TopicPartition
import base64
import json
import cv2
import numpy as np
import os
# Fire up the Kafka Consumer
topic = "video"

# Set the consumer in a Flask App
app = Flask(__name__)

# route to display stream of one camera
@app.route("/sdd/<int:cam_num>")
def sdd(cam_num):
    """
    This is the heart of our video display. Notice we set the mimetype to 
    multipart/x-mixed-replace. This tells Flask to replace any old images with 
    new values streaming through the pipeline.
    """
    consumer = KafkaConsumer(bootstrap_servers = ['localhost:9092'],
                             value_deserializer=lambda value: json.loads(value.decode()))
                                                                           
    consumer.assign([TopicPartition(topic = topic, partition = cam_num-1)])

    return Response(
        get_sdd_stream(consumer), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

def get_sdd_stream(consumer):
    """
    Here is where we recieve streamed images from the Kafka Server and convert 
    them to a Flask-readable format.
    """
    labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    countFrames = 0

    for msg in consumer:
        if countFrames%15 == 0:
            prediction_obj = msg.value
            string = prediction_obj['frame']
            jpg_original = base64.b64decode(string)
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            img = cv2.imdecode(jpg_as_np, flags=1)
            processedFrame = detect(img,net,ln,LABELS)
            ret,encodedFrame = cv2.imencode('.jpg',processedFrame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + encodedFrame.tobytes() + b'\r\n\r\n')
        countFrames += 1

# route to display stream of multiple cameras
# @app.route("/sdds/<int:camera_numbers>")
# def get_sdds(camera_numbers):
#     return render_template("videos.html", cam_nums = list(range(1,camera_numbers+1)))

@app.route("/pc/<int:cam_num>")
def pc(cam_num):
    """
    This is the heart of our video display. Notice we set the mimetype to 
    multipart/x-mixed-replace. This tells Flask to replace any old images with 
    new values streaming through the pipeline.
    """
    consumer = KafkaConsumer(bootstrap_servers = ['localhost:9092'],
                             value_deserializer=lambda value: json.loads(value.decode()))
                                                                           
    consumer.assign([TopicPartition(topic = topic, partition = cam_num-1)])

    return Response(
        get_pc_stream(consumer), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

def get_pc_stream(consumer):
    """
    Here is where we recieve streamed images from the Kafka Server and convert 
    them to a Flask-readable format.
    """
    for msg in consumer:
        prediction_obj = msg.value
        string = prediction_obj['frame']
        jpg_original = base64.b64decode(string)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
        processedFrame = count(img)
        ret,encodedFrame = cv2.imencode('.jpg',processedFrame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + encodedFrame.tobytes() + b'\r\n\r\n')

# @app.route("/pcs/<int:camera_numbers>")
# def get_pcs(camera_numbers):
#     return render_template("videos.html", cam_nums = list(range(1,camera_numbers+1)))

# @app.route("/copies/<int:cam_num>/<int:copies_num>")
# def get_copies(cam_num,copies_num):
#     return render_template("copies.html", cam_num = cam_num, copies_num = list(range(1,copies_num+1)))

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 3000, debug = True)
