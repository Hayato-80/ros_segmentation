#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import onnxruntime

model = '/home/jetson/catkin_ws/src/ros_segmentation/model/model.onnx'
session = onnxruntime.InferenceSession(model)

image = cv.imread('')
image = cv.resize(image,424,424)
image = cv.image.astype(np.float32)/255.0

image = np.expand_dims(images,axis=0)

ort_inputs = {ort_session.get_inputs()[0].name: image}
ort_outs = ort_session.run(None, ort_inputs)

mask = (ort_outputs[0]>0.5).astype(np.uint8)


result = cv.bitwise_and(image,image, mask=mask)

cv.imshow(result)
cv.waitKey(0)
cv.destroyAllWindows
