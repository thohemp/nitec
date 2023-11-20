import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torchvision
from torchvision import transforms
from dataset import NIT_EC
from tqdm import tqdm
from torchmetrics.classification import BinaryAveragePrecision
from nitec.model import ResNet
import torchvision.models as models

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)

    args = parser.parse_args()
    return args



def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define ResNet18 model; change model here if desired 
    model = ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 2) 
    # For ResNet50:
    #model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3],2)

    if not args.snapshot == '':
        # print("load snapshot...")
        saved_state_dict = torch.load(args.snapshot, map_location='cpu')
        if 'model_state_dict' in saved_state_dict:
            model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            model.load_state_dict(saved_state_dict)
        saved_state_dict = torch.load(args.snapshot)

    torch.save(model.state_dict(), "rs18_nitec.pth")
    model.to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([#transforms.Resize(224),       
                                        transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

    
    test_data = NIT_EC(transform=transformations, data_split='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tp0 = 0
    fp0 = 0
    tn0 = 0
    fn0 = 0
    fpr = []
    tpr = []
    prediction_list = []
    y_list = []
    model.eval()
    metric = BinaryAveragePrecision(thresholds=None)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            _, lbs = torch.max(labels, 1)

            prediction_list.append(outputs[0][1].item())
            y_list.append(int(lbs.item()))
            tp += torch.sum(preds == 1 and preds == lbs)
            fp += torch.sum(preds == 1 and preds != lbs )
            tn += torch.sum(preds == 0 and preds == lbs)
            fn += torch.sum(preds == 0 and preds != lbs)

            tp0 += torch.sum(preds == 0 and preds == lbs)
            fp0 += torch.sum(preds == 0 and preds != lbs )
            tn0 += torch.sum(preds == 1 and preds == lbs)
            fn0 += torch.sum(preds == 1 and preds != lbs)


        accuracy = (tp + tn) / (len(test_data)* 1.0)
        precision = (1.0 * tp) / (tp + fp)
        recall = (1.0 * tp) / (tp + fn)
        f1 = 2.0 / ((1.0 / precision) + (1.0 / recall))

        precision0 = (1.0 * tp0) / (tp0 + fp0)
        print(f"AP: { metric(torch.Tensor(prediction_list), torch.IntTensor(y_list))}")
        print(f"F1: { 2.0 / ((1.0 / precision) + (1.0 / recall))}")
