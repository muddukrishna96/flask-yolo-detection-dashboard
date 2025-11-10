import os
import numpy as np
import onnx
import onnxruntime as ort
import cv2


def _letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    # resize
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def _xywh_to_xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


def _nms(boxes, scores, iou_threshold=0.45):
    # boxes: Nx4 [x1,y1,x2,y2], scores: N
    if len(boxes) == 0:
        return []
    cv_boxes = [tuple(map(int, b)) for b in boxes]
    idxs = cv2.dnn.NMSBoxes(cv_boxes, [float(s) for s in scores], score_threshold=0.001, nms_threshold=iou_threshold)
    if len(idxs) == 0:
        return []
    flat = [i[0] if isinstance(i, (list, tuple)) else int(i) for i in idxs]
    return flat


class ONNXModelWrapper:
    """Thin wrapper over ONNXRuntime session exposing a YOLO-like callable API.
    __call__(image, save=False) -> list-like where [0].plot() returns plotted image.
    """
    def __init__(self, session: ort.InferenceSession, input_name: str, input_shape=(1,3,640,640)):
        self.session = session
        self.input_name = input_name
        self.input_shape = input_shape
        self.is_onnx = True

    def _run_inference(self, img: np.ndarray):
        h0, w0 = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        im_resized, ratio, (pad_x, pad_y) = _letterbox(img_rgb, (target_h, target_w))
        im_resized = im_resized.astype(np.float32) / 255.0
        im_trans = np.transpose(im_resized, (2, 0, 1))[None, :]
        ort_inputs = {self.input_name: im_trans}
        outs = self.session.run(None, ort_inputs)
        return outs, ratio, pad_x, pad_y, (w0, h0)

    def __call__(self, img: np.ndarray, save=False):
        outs, ratio, pad_x, pad_y, (w0, h0) = self._run_inference(img)
        dets = []
        out0 = outs[0]
        if out0.ndim == 3:
            _, N, C = out0.shape
            if C >= 5:
                for row in out0[0]:
                    x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    conf = float(row[4])
                    class_id = int(np.argmax(row[5:])) if C > 5 else 0
                    x1, y1, x2, y2 = _xywh_to_xyxy(x, y, w, h)
                    target_w = self.input_shape[3]
                    target_h = self.input_shape[2]
                    x1 = (x1 * target_w - pad_x) / ratio
                    y1 = (y1 * target_h - pad_y) / ratio
                    x2 = (x2 * target_w - pad_x) / ratio
                    y2 = (y2 * target_h - pad_y) / ratio
                    dets.append([x1, y1, x2, y2, conf, class_id])
        elif out0.ndim == 2 and out0.shape[1] >= 6:
            for row in out0:
                x1, y1, x2, y2, score, class_id = map(float, row[:6])
                dets.append([x1, y1, x2, y2, float(score), int(class_id)])
        else:
            dets = []

        boxes = [[d[0], d[1], d[2], d[3]] for d in dets]
        scores = [d[4] for d in dets]
        keep = _nms(boxes, scores) if boxes else []
        final = [dets[i] for i in keep]

        class DummyRes:
            def __init__(self, img_out):
                self._img = img_out

            def plot(self):
                return self._img

        img_out = img.copy()
        for d in final:
            x1, y1, x2, y2, conf, cls = d
            x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(w0 - 1, x2), min(h0 - 1, y2)])
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{int(cls)}:{conf:.2f}"
            cv2.putText(img_out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return [DummyRes(img_out)]


def validate_onnx_model(path: str, input_shape=(1, 3, 640, 640)):
    if not os.path.isfile(path):
        raise FileNotFoundError('Model file not found')

    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
    except Exception as e:
        raise RuntimeError(f'ONNX model failed validation: {e}')

    try:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    except Exception as e:
        raise RuntimeError(f'Failed to create ONNX runtime session: {e}')

    inputs = sess.get_inputs()
    if len(inputs) == 0:
        raise RuntimeError('ONNX model has no inputs')

    inp = inputs[0]
    input_name = inp.name

    try:
        dummy = np.zeros(input_shape, dtype=np.float32)
        out = sess.run(None, {input_name: dummy})
        if not isinstance(out, list) or len(out) == 0:
            raise RuntimeError('ONNX model produced no outputs')
    except Exception as e:
        raise RuntimeError(f'ONNX dummy inference failed: {e}')

    return ONNXModelWrapper(sess, input_name, input_shape)
