import time

import cv2
from PIL import Image
from ultralytics import YOLO

frame = cv2.imread('n.jpg')
model = YOLO('PUBG.pt')

results = model.predict(frame, task="detect", conf=0.4, device=0, verbose=False, imgsz=320)
time_start = time.perf_counter()

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # print("x1: {}, y1: {}, x2: {}, y2: {}".format(x1, y1, x2, y2))
        conf = box.conf.item()  # 置信度
        cls = int(box.cls.item())

        # 类别 0 代表人类
        # if cls == 0:
        #     max_conf_x1, max_conf_y1, max_conf_x2, max_conf_y2 = x1, y1, x2, y2
        #     print("x1: {}, y1: {}, x2: {}, y2: {}".format(max_conf_x1, max_conf_y1, max_conf_x2, max_conf_y2))
        #     cv2.rectangle(frame, (max_conf_x1, max_conf_y1), (max_conf_x2, max_conf_y2), (0, 0, 255), 2)
        #     cv2.putText(frame, f'Person: {conf:.2f}', (max_conf_x1, max_conf_y1 - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #     Image.fromarray(frame).show()
time_end = time.perf_counter()
time_consumed = time_end - time_start
print("cost: {} s".format(time_consumed))