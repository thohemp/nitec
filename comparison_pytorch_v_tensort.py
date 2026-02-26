import cv2
import numpy as np
import torch
import torchvision
import tensorrt as trt
from cuda import cudart
from pathlib import Path

from face_detection import RetinaFace
from nitec.model import ResNet
from nitec.utils import prep_input_numpy

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

class TRTModule:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

    def infer(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        self.context.set_input_shape(self.input_name, x.shape)
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
                raise RuntimeError("TRT execute failed")
            cudart.cudaMemcpy(out.ctypes.data, d_out, out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            return out
        finally:
            cudart.cudaFree(d_in)
            cudart.cudaFree(d_out)

# Paths
WEIGHTS = Path("models/nitec_rs18_e20.pth")
ENGINE  = Path("models/nitec_rs18_e20_fp32.engine")

assert WEIGHTS.exists(), f"Missing: {WEIGHTS}"
assert ENGINE.exists(),  f"Missing: {ENGINE}"

# 1) Load image and detect faces (same as demo)
img_path = "/content/nitec/images/5.jpeg"
frame = cv2.imread(img_path)  # img_path from upload step
assert frame is not None, "Could not read image"

detector = RetinaFace(gpu_id=0)
faces = detector(frame)

face_imgs = []
if faces is not None:
    for box, landmark, score in faces:
        if score < 0.7:
            continue
        x_min, y_min = max(int(box[0]), 0), max(int(box[1]), 0)
        x_max, y_max = int(box[2]), int(box[3])
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (224, 224))
        face_imgs.append(crop)

assert len(face_imgs) > 0, "No faces detected (try another image)."

# 2) Preprocess using repo function -> NCHW float32
device = torch.device("cuda")
batch_t = prep_input_numpy(np.stack(face_imgs), device)  # torch [B,3,224,224]
batch_np = batch_t.detach().cpu().numpy().astype(np.float32)

# 3) PyTorch inference (CPU is fine, but keep it consistent)
pt = ResNet(torchvision.models.resnet.BasicBlock, [2,2,2,2], 2)
ckpt = torch.load(WEIGHTS, map_location="cpu")
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
pt.load_state_dict(state)
pt.eval()

with torch.no_grad():
    pt_logits = pt(torch.from_numpy(batch_np)).numpy()      # [B,2]
pt_probs = sigmoid_np(pt_logits)[:, 1]                      # [B]

# 4) TensorRT inference
trt_model = TRTModule(str(ENGINE))
trt_logits = trt_model.infer(batch_np)                      # [B,2]
trt_probs = sigmoid_np(trt_logits)[:, 1]                    # [B]

# 5) Compare
ok = np.allclose(pt_probs, trt_probs, rtol=1e-3, atol=1e-4)
print("✅ np.allclose:", ok)
print("max_abs_err:", float(np.max(np.abs(pt_probs - trt_probs))))
print("pt_probs :", pt_probs)
print("trt_probs:", trt_probs)
