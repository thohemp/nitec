# <div align="center"> **NITEC: Versatile Hand-Annotated Eye Contact Dataset for Ego-Vision Interaction (WACV24)** </div>

<p align="center">
  <img src="https://github.com/thohemp/archive/blob/main/nitec.gif" alt="animated" />
</p>

---

## **Citing**

If you find our work useful, please cite the paper:

```BibTeX
@INPROCEEDINGS{10484276,
  author={Hempel, Thorsten and Jung, Magnus and Abdelrahman, Ahmed A. and Al-Hamadi, Ayoub},
  booktitle={2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={NITEC: Versatile Hand-Annotated Eye Contact Dataset for Ego-Vision Interaction}, 
  year={2024},
  pages={4425-4434},
  doi={10.1109/WACV57701.2024.00438}}
```

---

## <div align="center"> **Paper**</div>

> [Thorsten Hempel, Magnus Jung, Ahmed A. Abdelrahman and Ayoub Al-Hamadi, "NITEC: Versatile Hand-Annotated Eye Contact Dataset for Ego-Vision Interaction", *WACV 2024*.](https://openaccess.thecvf.com/content/WACV2024/papers/Hempel_NITEC_Versatile_Hand-Annotated_Eye_Contact_Dataset_for_Ego-Vision_Interaction_WACV_2024_paper.pdf)

---

## <div align="center"> **Abstract**</div>

> Eye contact is a crucial non-verbal interaction modality and plays an important role in our everyday social life. While humans are very sensitive to eye contact, the capabilities of machines to capture a person's gaze are still mediocre. We tackle this challenge and present NITEC, a hand-annotated eye contact dataset for ego-vision interaction. NITEC exceeds existing datasets for ego-vision eye contact in size and variety of demographics, social contexts, and lighting conditions, making it a valuable resource for advancing ego-vision-based eye contact research. Our extensive evaluations on NITEC demonstrate strong cross-dataset performance, emphasizing its effectiveness and adaptability in various scenarios, that allows seamless utilization to the fields of computer vision, human-computer interaction, and social robotics. We make our NITEC dataset publicly available to foster reproducibility and further exploration in the field of ego-vision interaction.

---

# <div align="center"> Quick Usage </div>

```sh
pip install face_detection@git+https://github.com/elliottzheng/face-detection
pip install nitec
```

Example usage:

```python
from nitec import NITEC_Classifier, visualize
import cv2
import torch
import pathlib

CWD = pathlib.Path.cwd()

nitec_pipeline = NITEC_Classifier(
    weights=CWD / 'models' / 'nitec_rs18_e20.pth',
    device=torch.device('cuda')  # or 'cpu'
)

cap = cv2.VideoCapture(0)

_, frame = cap.read()
results = nitec_pipeline.predict(frame)
frame = visualize(frame, results, confidence=0.5)
```

---

# <div align="center"> TensorRT Workflow (PyTorch → ONNX → TensorRT) </div>

This section documents how the pretrained NITEC model  
(`nitec_rs18_e20.pth`) was ported from PyTorch to TensorRT.

---

## 1️⃣ ONNX Export (opset 17 + dynamic batch)

The PyTorch model is exported to ONNX with:

- Opset version: 17  
- Dynamic batch size  
- Input shape: `[B, 3, 224, 224]`

Example:

```bash
python tools/export_onnx.py
```

Output file:

```
models/nitec_rs18_e20_opset17.onnx
```

> Note: For TensorRT parser compatibility, the legacy PyTorch ONNX exporter (`dynamo=False`) was used when required.

---

## 2️⃣ TensorRT Engine Build (FP32)

Convert ONNX to TensorRT engine:

```bash
python tools/build_engine.py \
  --onnx models/nitec_rs18_e20_opset17_legacy.onnx \
  --engine models/nitec_rs18_e20_fp32.engine
```



Output:

```
models/nitec_rs18_e20_fp32.engine
```

---

## 3️⃣ TensorRT Inference



### Colab (Image-based demo)

```bash
python demo_trt_image.py
```

---

## 4️⃣ Validation (PyTorch vs TensorRT)

Compare PyTorch and TensorRT outputs:

```bash
python comparison_pytorch_v_tensort.py
```

Expected output:

```
np.allclose: True
```

This confirms numerical equivalence between PyTorch and TensorRT inference.

---

# <div align="center"> Train / Test </div>

## NITEC Dataset

Prepare the dataset as explained [here](data/README.MD).

## Snapshots

Download from here:  
https://drive.google.com/drive/folders/1zc6NZZ6yA4NJ52Nn0bgky1XpZs9Z0hSJ?usp=sharing

## Train

```bash
python train.py \
 --gpu 0 \
 --num_epochs 50 \
 --batch_size 64 \
 --lr 0.0001
```

## Test

```bash
python test.py \
 --snapshot models/nitec_rs18_20.pth \
 --gpu 0
```