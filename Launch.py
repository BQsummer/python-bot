import threading
from datetime import datetime
from math import sqrt
from random import randrange
import onnxruntime as ort

import dxcam
import pythoncom
from PIL import Image
import torch
import onnx
import cv2
import numpy as np
from ultralytics import YOLO
import time
import win32com.client
import os
import torch
import threading, queue, time

import interception
from consts import interception_filter_key_state, interception_filter_mouse_state, interception_mouse_flag, \
    interception_mouse_state
from stroke import key_stroke, mouse_stroke



# config
area = 256
mode = 0 # 0: ai + recoil, 1: only ai, 2: only recoil
max_dist = area * area // 4
resolution_x = 2560
resolution_y = 1440
conf_c = 0.4
base_recoil = 1
max_conf_x1, max_conf_y1, max_conf_x2, max_conf_y2 = 0, 0, 0, 0
person_cls = 0

debug_show = True      # 是否imshow
debug_draw = True      # 是否画框
debug_pause = False    # 是否暂停
debug_save_dir = "dbg_caps"
os.makedirs(debug_save_dir, exist_ok=True)


smooth_dx, smooth_dy = 0.0, 0.0
last_dx, last_dy = 0, 0

SMOOTH = 0.1      # 越大越稳但越慢
MAX_MOVE = 50       # 原来20，先降
MAX_DELTA = 3      # 每帧变化限制
AXIS_DEAD = 2      # 单轴死区

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.enable_mem_pattern = True
so.enable_mem_reuse = True
so.intra_op_num_threads = 1  # 实时场景通常别开太多
so.inter_op_num_threads = 1
so.log_severity_level = 3    # 关 warning（不影响性能，只是安静）

cuda_opts = {
    "cudnn_conv_algo_search": "HEURISTIC",  # 或 "DEFAULT"
    "arena_extend_strategy": "kNextPowerOfTwo",
    "do_copy_in_default_stream": 1,         # 有时对延迟更友好
}

frame_q = queue.Queue(maxsize=1)
stop_flag = False


def capture_loop():
    global stop_flag
    while not stop_flag:
        f = camera.grab(region=region)
        if f is None:
            continue

        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)

        # 只保留最新帧
        if frame_q.full():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put_nowait(f)


def letterbox(im, new_shape=(area, area), color=(114, 114, 114)):
    """Resize + pad to new_shape, keep ratio. Return: padded_img, ratio, (dw, dh)."""
    h, w = im.shape[:2]
    new_w, new_h = new_shape[0], new_shape[1]

    r = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * r)), int(round(h * r))

    im_resized = cv2.resize(im, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    dw = new_w - resized_w
    dh = new_h - resized_h
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)

def nms_opencv_xyxy(boxes_xyxy: np.ndarray,
                    scores: np.ndarray,
                    conf_thres=0.4,
                    iou_thres=0.45) -> np.ndarray:
    """Return keep indices."""
    if boxes_xyxy.shape[0] == 0:
        return np.array([], dtype=np.int64)

    # cv2.dnn.NMSBoxes expects [x, y, w, h]
    b = boxes_xyxy.astype(np.float32).copy()
    b[:, 2] = b[:, 2] - b[:, 0]
    b[:, 3] = b[:, 3] - b[:, 1]

    idxs = cv2.dnn.NMSBoxes(
        b.tolist(),
        scores.astype(float).tolist(),
        conf_thres,
        iou_thres
    )
    if idxs is None or len(idxs) == 0:
        return np.array([], dtype=np.int64)
    return np.array(idxs).reshape(-1).astype(np.int64)

def nms_xyxy(boxes, scores, iou_thres=0.45):
    """Pure numpy NMS. boxes: (N,4) xyxy."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1e-9) * (y2 - y1 + 1e-9)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


# ----------------------------
# ONNX Runtime YOLOv8 wrapper
# ----------------------------
class YOLOv8ONNX:
    def __init__(self, onnx_path: str, use_cuda=True):
        providers = ort.get_available_providers()

        if use_cuda and "CUDAExecutionProvider" in providers:
            self.sess = ort.InferenceSession(
                onnx_path,
                sess_options=so,
                providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
            )
            self.on_gpu = True
        else:
            self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
            self.on_gpu = False

        self.input = self.sess.get_inputs()[0]
        self.input_name = self.input.name
        self.input_shape = self.input.shape

        self.output = self.sess.get_outputs()[0]
        self.output_name = self.output.name

        # IOBinding（每次复用，减少开销）
        self.io = self.sess.io_binding()

    def preprocess(self, frame_bgr, imgsz=area):
        # 如果 frame 已经是 (area, area)，直接走最短路径
        if frame_bgr.shape[0] == imgsz and frame_bgr.shape[1] == imgsz:
            img = frame_bgr.astype(np.float32, copy=False)
            img *= (1.0 / 255.0)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)
            # r=1, pad=0
            return img, 1.0, 0.0, 0.0

        # fallback：不是同尺寸才 letterbox
        img, r, (padw, padh) = letterbox(frame_bgr, (imgsz, imgsz))
        img = img.astype(np.float32, copy=False)
        img *= (1.0 / 255.0)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img, r, padw, padh

    def infer(self, frame_bgr, conf_thres=0.4, iou_thres=0.35, imgsz=256):
        orig_h, orig_w = frame_bgr.shape[:2]
        inp, r, padw, padh = self.preprocess(frame_bgr, imgsz=imgsz)

        # 绑定输入输出（减少内部拷贝/同步）
        self.io.clear_binding_inputs()
        self.io.clear_binding_outputs()

        # 输入：CPU numpy -> ORT（ORT 会负责把它送到 CUDA；如果你想更极致，可以再做 pinned memory）
        self.io.bind_cpu_input(self.input_name, inp)

        # 输出：让 ORT 自己分配（也可以 bind 到固定 buffer）
        self.io.bind_output(self.output_name)

        self.sess.run_with_iobinding(self.io)
        preds = self.io.copy_outputs_to_cpu()[0]

        boxes, scores, clses = self.postprocess_yolov8(
            preds, r, padw, padh, orig_w, orig_h,
            conf_thres=conf_thres, iou_thres=iou_thres
        )
        return boxes, scores, clses

    def postprocess_yolov8(self, preds, r, padw, padh, orig_w, orig_h,
                           conf_thres=0.4, iou_thres=0.45):
        pred = preds
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] in (84, 85, 4 + 80):
            pred = pred.transpose(1, 0)

        # 只取 person 类的 score（避免 argmax / cls_id 数组）
        xywh = pred[:, :4]
        score = pred[:, 4 + person_cls]

        m = score >= conf_thres
        if not np.any(m):
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int64))

        xywh = xywh[m]
        score = score[m]

        # xywh -> xyxy
        x = xywh[:, 0];
        y = xywh[:, 1];
        w = xywh[:, 2];
        h = xywh[:, 3]
        boxes = np.empty((xywh.shape[0], 4), dtype=np.float32)
        boxes[:, 0] = x - w * 0.5
        boxes[:, 1] = y - h * 0.5
        boxes[:, 2] = x + w * 0.5
        boxes[:, 3] = y + h * 0.5

        # unpad + unscale
        boxes[:, [0, 2]] -= padw
        boxes[:, [1, 3]] -= padh
        boxes *= (1.0 / r)

        # clip（原地）
        np.clip(boxes[:, 0], 0, orig_w - 1, out=boxes[:, 0])
        np.clip(boxes[:, 1], 0, orig_h - 1, out=boxes[:, 1])
        np.clip(boxes[:, 2], 0, orig_w - 1, out=boxes[:, 2])
        np.clip(boxes[:, 3], 0, orig_h - 1, out=boxes[:, 3])

        keep = nms_opencv_xyxy(boxes, score, conf_thres=conf_thres, iou_thres=iou_thres)
        boxes = boxes[keep]
        scores = score[keep]
        clses = np.full((len(keep),), person_cls, dtype=np.int64)
        return boxes, scores, clses

# ----------------------------
# Create model once (DON'T create every frame)
# ----------------------------
ort_model = YOLOv8ONNX("256.onnx", use_cuda=True)
print("onnxruntime providers:", ort_model.sess.get_providers(), "on_gpu:", ort_model.on_gpu)


def detect_people(frame_p, return_debug=False):
    """
    ROI(frame_p) -> target dict
    return_debug=True 时：额外返回 (boxes_person, scores_person, chosen_index, aim_point)
    """
    boxes, scores, clses = ort_model.infer(frame_p, conf_thres=conf_c, iou_thres=0.35, imgsz=area)

    #print("unique clses:", np.unique(clses), "max score:", scores.max() if len(scores) else None)
    idx = np.where(clses == person_cls)[0]
    if idx.size == 0:
        empty = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        if return_debug:
            return empty, np.zeros((0,4),dtype=np.float32), np.zeros((0,),dtype=np.float32), None, None
        return empty

    b = boxes[idx]
    s = scores[idx]

    # 选离中心最近
    cx = (b[:, 0] + b[:, 2]) / 2.0
    cy = (b[:, 1] + b[:, 3]) / 2.0
    target_x = area / 2.0
    target_y = area / 2.0
    dist = (cx - target_x) ** 2 + (cy - target_y) ** 2
    k = int(np.argmin(dist))

    x1, y1, x2, y2 = b[k].astype(int).tolist()

    # 你用于瞄准的点：dy 使用 y1 + 0.75*h
    aim_x = int((x1 + x2) * 0.5)
    aim_y = int((y2 - y1) * 0.75 + y1)
    target = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    #print("boxes are:", b)

    if return_debug:
        return target, b, s, k, (aim_x, aim_y)
    return target

def draw_debug(frame_bgr, boxes=None, scores=None, clses=None, target_idx=None, aim_point=None):
    """
    frame_bgr: ROI图 (area x area)
    boxes: (N,4) xyxy
    target_idx: 选中的目标索引（在boxes数组里）
    aim_point: (ax, ay) 你用于瞄准的点
    """
    vis = frame_bgr.copy()

    # 画准星中心
    cx, cy = area // 2, area // 2
    cv2.drawMarker(vis, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)

    if boxes is not None and len(boxes) > 0 and debug_draw:
        for i, b in enumerate(boxes.astype(int)):
            x1, y1, x2, y2 = b.tolist()
            thick = 3 if (target_idx is not None and i == target_idx) else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), thick)

            if scores is not None:
                conf = float(scores[i])
                cv2.putText(vis, f"{i}:{conf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 画“你实际用于计算dy的点”
    if aim_point is not None and debug_draw:
        ax, ay = aim_point
        cv2.circle(vis, (int(ax), int(ay)), 4, (0, 255, 255), -1)
        cv2.putText(vis, "aim", (int(ax) + 6, int(ay) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return vis

def handle_debug_keys(win_name, vis_img):
    """
    返回：是否退出
    按键：
      q/ESC: 退出
      p: 暂停/继续
      n: 暂停时单步
      d: 开关画框
      s: 保存当前帧
    """
    global debug_pause, debug_draw
    key = cv2.waitKey(1 if not debug_pause else 0) & 0xFF

    if key in (27, ord('q')):
        return True

    if key == ord('p'):
        debug_pause = not debug_pause

    if key == ord('d'):
        debug_draw = not debug_draw

    if key == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(debug_save_dir, f"{ts}.png")
        cv2.imwrite(path, vis_img)
        print("saved:", path)

    if debug_pause and key == ord('n'):
        # 单步：直接返回False，让主循环继续跑一帧
        debug_pause = True

    return False


min_d = 5
max_d = 80

def get_movement(p):
    if p["x1"] == 0 and p["y1"] == 0 and p["x2"] == 0 and p["y2"] == 0:
        return 0, 0
    dx = -area//2 + int((p["x1"] + p["x2"]) * 0.5)
    dy = -area//2 + int((p["y2"] - p["y1"]) * 0.3 + p["y1"])
    # abs dx + abs dy < 5
    dp = dx ** 2 + dy ** 2
    if dp < min_d ** 2:
        return 0, 0
    if dp > max_d ** 2:
        return int(dx * max_d // sqrt(dp)), int(dy * max_d // sqrt(dp))
    return dx, dy




def add_listener():
    global on, mouse_down, mouse_down_right
    while True:
        de = c.wait()
        stroke = c.receive(de)
        c.send(de, stroke)
        if type(stroke) is key_stroke:
            # print(stroke.code)
            if stroke.code == 25 and stroke.state == 1:
                on = not on
                mouse_down = False
                print("switch: " + str(on))
            continue
        if type(stroke) is mouse_stroke:
            # print(str(stroke.state) + " " + " " + str(stroke.x) + " " + str(stroke.y) + " " + str(stroke.flags))
            if stroke.state == 1:
                # print("switch 2: " + str(on))
                mouse_down = True
                continue
            if stroke.state == 2:
                # print("switch 3: " + str(on))
                mouse_down = False
                continue
            # if stroke.state == 3:
            #     #print("switch 2: " + str(on))
            #     mouse_down_right = True
            #     continue
            # if stroke.state == 4:
            #     #print("switch 3: " + str(on))
            #     mouse_down_right = False
            #     continue


def switch_level():
    """
    recoil level switch
    """
    global level
    if level == 5:
        level = 0
    else:
        level += 1

def warning():
    """
    speak to warn about current config
    :return:
    """
    global on, last_speak, speaker, find, level
    now = time.time()
    if speaker is None:
        pythoncom.CoInitialize()
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        pythoncom.CoInitialize()
    if on and find and now - last_speak > 2:
        last_speak = now
        speaker.Speak("锁")
    if on and (mode == 0 or mode == 2) and now - last_speak > 10:
        last_speak = now
        speaker.Speak(level)


def mouse_move_relative(x, y):
    c.send(device, mouse_stroke(0, interception_mouse_flag.INTERCEPTION_MOUSE_MOVE_RELATIVE.value, 0, x, y, 0))


def auto_recoil():
    global find
    while True:
        if on and (mode == 0 or mode == 2) and mouse_down:
            # mouse_move_relative(0, int(base_recoil * (1 + level * 0.1)))
            mouse_move_relative(0, 4)
            time.sleep(0.03)



# sign
on = False
dev = True
last_speak = 0
mouse_down = False
mouse_down_right = False
find = False
# recoil is 5~15, level count is 6 (0~5)
level = 0

# device = c.wait()
device = 11
print("device = {}".format(device))

# interception init
c = interception.interception()
c.set_filter(c.is_keyboard, interception_filter_key_state.INTERCEPTION_FILTER_KEY_UP.value | interception_filter_key_state.INTERCEPTION_FILTER_KEY_DOWN.value)
c.set_filter(c.is_mouse, interception_filter_mouse_state.INTERCEPTION_FILTER_MOUSE_LEFT_BUTTON_DOWN.value
             | interception_filter_mouse_state.INTERCEPTION_FILTER_MOUSE_LEFT_BUTTON_UP.value
             | interception_filter_mouse_state.INTERCEPTION_FILTER_MOUSE_RIGHT_BUTTON_DOWN.value
             | interception_filter_mouse_state.INTERCEPTION_FILTER_MOUSE_RIGHT_BUTTON_UP.value)

listener_thread = threading.Thread(target=add_listener)
listener_thread.daemon = True
listener_thread.start()


# recoil
threading.Thread(target=auto_recoil).start()


# warning
# pythoncom not support in a separate thread
speaker = None
speaker_thread = threading.Thread(target=warning)
speaker_thread.daemon = True
# speaker_thread.start()
if speaker is None:
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    pythoncom.CoInitialize()


# camera init
camera = dxcam.create()
# camera.start(target_fps=160)

left, top = (resolution_x - area) // 2, (resolution_y - area) // 2
right, bottom = left + area, top + area
region = (left, top, right, bottom)

print("left: {}, top: {}, right: {}, bottom: {}".format(left, top, right, bottom))


try:
    start = datetime.now()
    win_name = "ROI Debug"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    cap_th = threading.Thread(target=capture_loop, daemon=True)
    cap_th.start()

    filt_tx, filt_ty = None, None
    last_tx, last_ty = None, None
    last_print_t = 0.0

    while True:
        # print("\n--- New Loop ---")
        t0 = time.perf_counter()
        if not on or mode == 2:
            # 也可以在这里稍微sleep一下避免空转烧CPU
            # time.sleep(0.001)
            continue

        frame = frame_q.get()
        t1 = time.perf_counter()
        #print("grab cost:", t1 - t0)

        if frame is None:
            find = False
            continue

        # if frame.shape[2] == 4:
        #     # 大概率是 BGRA 或 RGBA；先按 BGRA 处理（dxcam常见）
        #     print( "frame has 4 channels, convert BGRA->BGR")
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # else:
        #     # 如果你现在看到红蓝反，那说明 frame 实际是 RGB
        #     print( "frame has 3 channels, convert RGB->BGR")
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        t2 = time.perf_counter()
        #print("convert cost:", t2 - t1)

        # --- detect + debug data ---
        position, p_boxes, p_scores, chosen_k, aim_pt = detect_people(frame, return_debug=True)
        t3= time.perf_counter()
        #print("detect cost:", t3 - t2)

        # --- show window ---
        # if debug_show:
        #     if debug_show:
        #         vis = draw_debug(frame, boxes=p_boxes, scores=p_scores,
        #                          target_idx=chosen_k, aim_point=aim_pt)
        #
        #         cv2.imshow(win_name, vis)
        #
        #         key = cv2.waitKey(1) & 0xFF  # ⭐关键：非阻塞刷新
                # if handle_debug_keys(key):
                #     break

        movement = get_movement(position)
        if movement[0] == 0 and movement[1] == 0:
            find = False
            continue

        if on and mouse_down:
            find = True
            # === 1. 统一瞄点（和 detect_people 保持一致） ===
            cx = area // 2
            cy = area // 2

            tx = int((position["x1"] + position["x2"]) * 0.5)
            ty = int((position["y2"] - position["y1"]) * 0.5 + position["y1"])  # 和 detect_people 一致

            dx = tx - cx
            dy = ty - cy

            # === 2. 单轴死区（防止微抖） ===
            if abs(dx) <= AXIS_DEAD:
                dx = 0
            if abs(dy) <= AXIS_DEAD:
                dy = 0

            if dx == 0 and dy == 0:
                continue

            # === 3. 距离相关比例增益（远快近稳） ===
            dist2 = dx * dx + dy * dy
            if dist2 > 40 * 40:
                k = 0.85
            elif dist2 > 20 * 20:
                k = 0.6
            else:
                k = 0.35

            dx = int(dx * k)
            dy = int(dy * k)

            # === 4. 对称限幅（非常重要，防止过冲） ===
            dx = max(-MAX_MOVE, min(MAX_MOVE, dx))
            dy = max(-MAX_MOVE, min(MAX_MOVE, dy))

            # === 5. 执行移动 ===
            if dx != 0 or dy != 0:
                mouse_move_relative(dx, dy)
            #print(f"move dx: {dx}, dy: {dy}")

            t4 = time.perf_counter()
            #print("move cost: ", t4 - t3)
            #print("loop cost:", t4 - t0)
finally:
    pass