import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torchvision
from torchvision import transforms
import random
import os
from dataset import NIT_EC
import time
from tqdm import tqdm
from nitec.model import ResNet
from collections import OrderedDict


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=50, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=64, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0001, type=float)
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)

    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':
    set_seed(2022)

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    summary_name = '{}_{}_bs{}'.format('NITEC', int(time.time()), args.batch_size)

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 2)
  
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot) #.jit
        print("load snapshot...")
        if 'model_state_dict' in saved_state_dict:
            model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            model.load_state_dict(saved_state_dict)

    model.to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

    train_data = NIT_EC(transform=transformations, data_split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    for epoch in range(num_epochs):
      
        # Training loop
        model.train()
        with tqdm(train_loader) as _tqdm:
            _tqdm.set_description('Epoch {}'.format(epoch+1))

            train_loss = 0.0
            train_acc = 0.0

            for i, (inputs, labels) in enumerate(_tqdm):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
            
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, lbs = torch.max(labels, 1)
                _, preds = torch.max(outputs, 1)
                train_acc += torch.sum(preds == lbs)
                                    
                _tqdm.set_postfix(OrderedDict(loss=f'{loss.item():.3f}'))

   
        train_loss = train_loss / len(train_data)
        train_acc = train_acc / len(train_data)
        
        
        # Print progress
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc))

        
        if not os.path.exists('output/snapshots/{}'.format(summary_name)):
            print("Create folder: output/snapshots/{}".format(summary_name))
            os.makedirs('output/snapshots/{}'.format(summary_name))

            # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                      torch.save(model.state_dict(), 'output/snapshots/' + summary_name + '/' + args.output_string +
                      '_epoch_' + str(epoch+1) + '.pth')
                  )
        