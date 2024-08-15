import cv2 as cv


def draw_bbox(image, xyxy, text, color=(0, 0, 255)):
    x_min = xyxy[0]
    y_min = xyxy[1]
    x_max = xyxy[2]
    y_max = xyxy[3]
    start, end = (x_min, y_min), (x_max, y_max)
    cv.rectangle(image, start, end, color, 1)
    cv.putText(image, text, (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv.LINE_AA)
    return image

def detector(file_name, model):
    img = cv.imread(f'data/images/{file_name}')
    img = cv.resize(img, (640, 640))
    results = model.predict(img, classes=[2])
    for box_index in results[0].boxes.xyxy:
      bbox_yolo = box_index.numpy().astype(int)
      draw_bbox(img, bbox_yolo ,'YOLOv8m', (0, 255, 0))
    return img

def detector_img(img, model):
    #img = cv.imread(img)
    #img = cv.resize(img, (640, 640))
    results = model.predict(img, classes=[2])
    for box_index in results[0].boxes.xyxy:
      bbox_yolo = box_index.numpy().astype(int)
      draw_bbox(img, bbox_yolo ,'YOLOv8m', (255, 255, 0))
    return img




