import os
import cv2 as cv
import numpy as np

folder_to_models = '/Users/kuanyshbakytuly/Desktop/Relive/2d_api/models/face_detector'

proto_path = os.path.sep.join([f"{folder_to_models}/deploy.prototxt"])
model_path = os.path.sep.join([f"{folder_to_models}/res10_300x300_ssd_iter_140000.caffemodel"])
face_detector = cv.dnn.readNetFromCaffe(proto_path, model_path)

def face_detection(image):
    (h,w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    face = np.zeros((3, 224, 224))

    if len(detections)>0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY-10:endY+10, startX-10:endX+10]
            return True, face
    return False, [], 