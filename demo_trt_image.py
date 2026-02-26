import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorrt as trt
from cuda import cudart

import torch
from face_detection import RetinaFace

from nitec.utils import prep_input_numpy
from nitec.visualize import visualize
from nitec.results import NITECResultContainer


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TRTModule:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to load engine.")
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)   # images
        self.output_name = self.engine.get_tensor_name(1)  # logits

    def infer(self, x: np.ndarray) -> np.ndarray:
        # x: [B,3,224,224] float32
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        b, c, h, w = x.shape
        self.context.set_input_shape(self.input_name, (b, c, h, w))

        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out = np.empty(out_shape, dtype=np.float32)

        err, d_in = cudart.cudaMalloc(x.nbytes);  assert err == 0
        err, d_out = cudart.cudaMalloc(out.nbytes); assert err == 0

        try:
            cudart.cudaMemcpy(d_in, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            self.context.set_tensor_address(self.input_name, int(d_in))
            self.context.set_tensor_address(self.output_name, int(d_out))

            ok = self.context.execute_async_v3(0)
            if not ok:
                raise RuntimeError("TensorRT execution failed.")

            cudart.cudaMemcpy(out.ctypes.data, d_out, out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            return out
        finally:
            cudart.cudaFree(d_in)
            cudart.cudaFree(d_out)


def run_trt_on_image(image_path: str, engine_path: str, conf_det: float = 0.7, conf_vis: float = 0.5):
    # Read image (BGR)
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Detector + TRT
    detector = RetinaFace(gpu_id=0)
    trt_model = TRTModule(engine_path)
    device = torch.device("cuda")

    faces = detector(frame)

    face_imgs, bboxes, landmarks, scores = [], [], [], []

    if faces is not None:
        for box, landmark, score in faces:
            if score < conf_det:
                continue

            x_min, y_min = max(int(box[0]), 0), max(int(box[1]), 0)
            x_max, y_max = int(box[2]), int(box[3])

            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (224, 224))

            face_imgs.append(crop)
            bboxes.append(box)
            landmarks.append(landmark)
            scores.append(score)

    if len(face_imgs) == 0:
        # nothing detected; just show original
        out_bgr = frame
        return out_bgr, None

    # preprocess (repo function) -> NCHW float
    batch_t = prep_input_numpy(np.stack(face_imgs), device)  # torch tensor [B,3,224,224]
    batch = batch_t.detach().cpu().numpy().astype(np.float32)

    logits = trt_model.infer(batch)       # [B,2]
    probs = sigmoid(logits)[:, 1]         # [B]

    results = NITECResultContainer(
        results=probs,
        bboxes=np.stack(bboxes),
        landmarks=np.stack(landmarks),
        scores=np.stack(scores),
    )

    out_bgr = visualize(frame.copy(), results, confidence=conf_vis)
    return out_bgr, probs


# ---- RUN ----
ENGINE_PATH = "models/nitec_rs18_e20_fp32.engine"  # change if your engine name differs
img_path = "/content/nitec/images/5.jpeg"
out_bgr, probs = run_trt_on_image(img_path, ENGINE_PATH)

# show result in colab
out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 7))
plt.imshow(out_rgb)
plt.axis("off")
plt.show()

if probs is not None:
    print("Face scores (class-1 prob):", probs)
