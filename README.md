# <div align="center"> **NITEC: Versatile Hand-Annotated Eye Contact Dataset for Ego-Vision Interaction (Accepted at WACV24)** </div>

<p align="center">
  <img src="https://github.com/thohemp/archive/blob/main/nitec.gif" alt="animated" />
</p>

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
## <div align="center"> **Paper**</div>
> [Thorsten Hempel, Magnus Jung, Ahmed A. Abdelrahman and Ayoub Al-Hamadi, "NITEC: Versatile Hand-Annotated Eye Contact Dataset for Ego-Vision Interaction", *accepted at WACV 2024*.](https://arxiv.org/abs/2311.04505)

## <div align="center"> **Abstract**</div>
>Eye contact is a crucial non-verbal interaction modality and plays an important role in our everyday social life. While humans are very sensitive to eye contact, the capabilities of machines to capture a person's gaze are still mediocre. We tackle this challenge and present NITEC, a hand-annotated eye contact dataset for ego-vision interaction. NITEC exceeds existing datasets for ego-vision eye contact in size and variety of demographics, social contexts, and lighting conditions, making it a valuable resource for advancing ego-vision-based eye contact research. Our extensive evaluations on NITEC demonstrate strong cross-dataset performance, emphasizing its effectiveness and adaptability in various scenarios, that allows seamless utilization to the fields of computer vision, human-computer interaction, and social robotics. We make our NITEC dataset publicly available to foster reproducibility and further exploration in the field of ego-vision interaction.


#  <div align="center"> Quick Usage: </div>

```sh
pip install face_detection@git+https://github.com/elliottzheng/face-detection
pip install nitec
```

Example usage:

```py
from nitec import NITEC_Classifier, visualize
import cv2

nitec_pipeline = NITEC_Classifier(
    weights= CWD / 'models' / 'nitec_rs18_e20.pth',
    device=torch.device('cuda') # or 'cpu'
)

cap = cv2.VideoCapture(0)

_, frame = cap.read()    
# Process frame and visualize
results = nitec_pipeline.predict(frame)
frame = visualize(frame, results, confidence=0.5)

```



# <div align="center">  Train / Test </div>

## NITEC Dataset
Prepare the dataset as explained [ here](data/README.MD).

## Snapshots

Download from here: https://drive.google.com/drive/folders/1zc6NZZ6yA4NJ52Nn0bgky1XpZs9Z0hSJ?usp=sharing

## Train
```py
 python train.py \
 --gpu 0 \
 --num_epochs 50 \
 --batch_size 64 \
 --lr 0.0001 \
```


## Test

```py
 python test.py \
 --snapshot models/nitec_rs18_20.pth \
 --gpu 0 \
```

