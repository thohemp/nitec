import torch, torchvision
from pathlib import Path
from nitec.model import ResNet

WEIGHTS = Path("models/nitec_rs18_e20.pth")
ONNX17_LEGACY = Path("models/nitec_rs18_e20_opset17_legacy.onnx")

model = ResNet(torchvision.models.resnet.BasicBlock, [2,2,2,2], 2)
ckpt = torch.load(WEIGHTS, map_location="cpu")
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1,3,224,224, dtype=torch.float32)

torch.onnx.export(
    model,
    dummy,
    ONNX17_LEGACY.as_posix(),
    opset_version=17,
    input_names=["images"],
    output_names=["logits"],
    dynamic_axes={"images": {0:"batch"}, "logits": {0:"batch"}},
    do_constant_folding=True,
    dynamo=False,  # ✅ KEY: legacy exporter
)

print("✅ Exported:", ONNX17_LEGACY)