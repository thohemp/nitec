
import numpy as np
from torchvision import transforms
import torch

transformations = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



def prep_input_numpy(img:np.ndarray, device:str):
    """Preparing a Numpy Array as input to the NITEC classifier."""

    if len(img.shape) == 4:
        imgs = []
        for im in img:
            imgs.append(transformations(im))
        img = torch.stack(imgs)
    else:
        img = transformations(img)

    img = img.to(device)

    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    return img
