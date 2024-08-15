from image_detect import detector
import cv2 as cv
import pandas as pd
from ultralytics import YOLO

df = pd.read_csv('data/labels.csv')



import os
folder_path = 'data/images/'
file_names = []
for file_name in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, file_name)):
        file_names.append(file_name)


# for file_name in file_names:
#     detector(file_name)

print(file_names)
#detector('vid_4_620.jpg', model)
model = YOLO('yolov8n.pt')

for file_name in file_names:
    cv.imwrite(f'data/marked_images/{file_name}', detector(file_name, model))