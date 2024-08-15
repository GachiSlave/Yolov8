from fastapi import FastAPI
import io
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from  image_detect import  detector_img
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# with open("data/images/vid_4_600.jpg", "rb") as f:
#     im_b64 = base64.b64encode(f.read())

with open("base64img.txt", "r") as f:
    base64_img = f.read()
def stringToImage(base64_string):
    im_bytes = base64.b64decode(base64_string)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return Image.fromarray(img)

def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

_, im_arr = cv2.imencode('.jpg', detector_img(toRGB(stringToImage(base64_img)), model))
im_bytes = im_arr.tobytes()
im_b64 = base64.b64encode(im_bytes)
print(im_b64)

