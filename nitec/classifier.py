import pathlib
from typing import Union
import torchvision

import cv2
import numpy as np
import torch
import torch.nn as nn
from face_detection import RetinaFace
import torch.nn.functional as F
import os 

from .results import NITECResultContainer
from .model import ResNet
from .utils import prep_input_numpy


class NITEC_Classifier:

    def __init__(
        self, 
        weights: pathlib.Path, 
        device: str = 'cpu', 
        include_detector:bool = True,
        confidence_threshold:float = 0.5
        ):

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Create NITEC model
        self.model = ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 2)
        saved_state_dict = torch.load(os.path.join(self.weights), map_location=device)
        if 'model_state_dict' in saved_state_dict:
            self.model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(saved_state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Create RetinaFace if requested
        if self.include_detector:

            if device.type == 'cpu':
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=device.index)

        
   
    def predict(self, frame: np.ndarray) -> NITECResultContainer:

        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []
        results = np.empty((0,1))
        bboxes_stack = np.empty((0,1))
        landmarks_stack = np.empty((0,1))
        scores_stack = np.empty((0,1))

        if self.include_detector:
            faces = self.detector(frame)
            if faces is not None: 
                for box, landmark, score in faces:

                    # Apply threshold
                    if score < .7:
                        continue
                               
                    # Extract safe min and max of x,y
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    
                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)
            
                # Classify eye contact
                if len(face_imgs):
                    results = self.process_frame(np.stack(face_imgs))
                    bboxes_stack = np.stack(bboxes)
                    landmarks_stack = np.stack(landmarks)
                    scores_stack = np.stack(scores)
            else:

                results = np.empty((0,1))

        else:
            results = self.process_frame(frame)

        # Save data
        results = NITECResultContainer(
            results=results,
            bboxes=bboxes_stack,
            landmarks=landmarks_stack,
            scores=scores_stack
        )

        return results

    def process_frame(self, frame: Union[np.ndarray, torch.Tensor]):
        
        # Prepare input
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise RuntimeError("Invalid dtype for input")
    
        # Classify
        output = self.model(img)
        output = F.sigmoid(output)
        val = output.cpu().numpy()[:,1]
        return val

