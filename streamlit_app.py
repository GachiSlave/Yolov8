import json

import streamlit as st
from image_detect import detector_img
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import requests



st.title("Детектор машинок, при загрузки файла он показывает картинку с отмеченной границей машины")
model = YOLO('yolov8n.pt')


uploaded_file  = st.file_uploader("Загрузите картинку JPG или PNG", ['png','jpg'])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    st.image(detector_img(img , model))

st.title("При нажатии кнопки отправляется post запрос к FastAPI")
json64 ={}
json64['str64'] = st.text_input("Введите текст:")

button_submit = st.button("отправка")

if button_submit:
    response = requests.post(url = 'http://127.0.0.1:8000/predict', data=json.dump(json64))
    st.success(response.text)