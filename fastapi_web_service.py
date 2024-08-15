from fastapi import FastAPI
import cv2
import base64
import numpy as np
from PIL import Image
from pydantic import BaseModel
from  image_detect import  detector_img
from ultralytics import YOLO


class Base64_string(BaseModel):
    base64_string: str

model = YOLO('yolov8n.pt')

app = FastAPI()

with open("data/images/vid_4_600.jpg", "rb") as f:
    im_b64 = base64.b64encode(f.read())
def stringToImage(base64_string):
    im_bytes = base64.b64decode(base64_string)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return Image.fromarray(img)

def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)



@app.post("/pedict")
async def predict(item: Base64_string):
    img = detector_img(toRGB(stringToImage(item.base64_string)), model)
    _, im_arr = cv2.imencode('.jpg', img)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64



