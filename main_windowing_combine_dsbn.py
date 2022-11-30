'''Train Office31 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets

import os
import argparse

from apn_dsbn_resnext import APN

import random
from PIL import Image
import numpy as np
from itertools import cycle
import math
import pickle

from util import ImageFolder2

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


parser = argparse.ArgumentParser(description='PyTorch SpineMets CT Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch_decay_start', type=int, default=60)
parser.add_argument('--kth', default=0, help='the kth run of the algorithm for the same seed')
parser.add_argument('--note', default='', type=str)

parser.add_argument('--fs', default=512, type=int)
parser.add_argument('--lamb', default=0.5, type=float)
parser.add_argument('--temp', default=10.0, type=float)


args = parser.parse_args()
store_weights = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc_Abdomen = 0  # best test accuracy
last_acc_Abdomen = 0
best_avg_acc_Abdomen = 0
last_avg_acc_Abdomen = 0

best_acc_Bone = 0  # best test accuracy
last_acc_Bone = 0
best_avg_acc_Bone = 0
last_avg_acc_Bone = 0

best_acc_Spine = 0  # best test accuracy
last_acc_Spine = 0
best_avg_acc_Spine = 0
last_avg_acc_Spine = 0


start_epoch = 0  # start from epoch 0 or last checkpoint epoch

nb_classes = 3
nb_epochs = args.n_epoch
batch_size = 24

transform_train_Abdomen = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.77036041,0.77036041,0.77036041),  (0.25962961,0.25962961,0.25962961)),
])

transform_test_Abdomen = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.77036041,0.77036041,0.77036041),  (0.25962961,0.25962961,0.25962961)),
])


transform_train_Bone = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.44330771,0.44330771,0.44330771),  (0.1424465,0.1424465,0.1424465)),
])

transform_test_Bone = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.44330771,0.44330771,0.44330771),  (0.1424465,0.1424465,0.1424465)),
])


transform_train_Spine = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.63147502,0.63147502,0.63147502),  (0.40628336,0.40628336,0.40628336)),
])

transform_test_Spine = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.63147502,0.63147502,0.63147502),  (0.40628336,0.40628336,0.40628336)),
])


train_dir_Abdomen = '/hdd8/zhulei/ct-bilsky-dataset/windowings-aug/Abdomen soft tissue window/trainval/train'
val_dir_Abdomen = '/hdd8/zhulei/ct-bilsky-dataset/windowings-aug/Abdomen soft tissue window/trainval/val'

train_dir_Bone = '/hdd8/zhulei/ct-bilsky-dataset/windowings-aug/Bone window/trainval/train'
val_dir_Bone = '/hdd8/zhulei/ct-bilsky-dataset/windowings-aug/Bone window/trainval/val'

train_dir_Spine = '/hdd8/zhulei/ct-bilsky-dataset/windowings-aug/Spine soft tissue window/trainval/train'
val_dir_Spine = '/hdd8/zhulei/ct-bilsky-dataset/windowings-aug/Spine soft tissue window/trainval/val'


folders = ['normal', '1', '2']

class_indexes_train_Abdomen = {}
for folder in folders:
    for file in os.listdir(os.path.join(train_dir_Abdomen, folder)):
        if folders.index(folder) in class_indexes_train_Abdomen:
            class_indexes_train_Abdomen[folders.index(folder)].append(os.path.join(train_dir_Abdomen, folder, file))
        else:
            class_indexes_train_Abdomen[folders.index(folder)] = [os.path.join(train_dir_Abdomen, folder, file)]

for k, v in class_indexes_train_Abdomen.items():
    v.sort()


class_indexes_val_Abdomen = {}
for folder in folders:
    for file in os.listdir(os.path.join(val_dir_Abdomen, folder)):
        if folders.index(folder) in class_indexes_val_Abdomen:
            class_indexes_val_Abdomen[folders.index(folder)].append(os.path.join(val_dir_Abdomen, folder, file))
        else:
            class_indexes_val_Abdomen[folders.index(folder)] = [os.path.join(val_dir_Abdomen, folder, file)]

for k, v in class_indexes_val_Abdomen.items():
    v.sort()


random.seed(args.seed)
for i in range(0, nb_classes):
    random.shuffle(class_indexes_train_Abdomen[i])

image_list_train_Abdomen = []
for k,v in class_indexes_train_Abdomen.items():
    for e in v:
        image_list_train_Abdomen.append((k,e))
random.shuffle(image_list_train_Abdomen)

for i in range(0, nb_classes):
    random.shuffle(class_indexes_val_Abdomen[i])

image_list_val_Abdomen = []
for k,v in class_indexes_val_Abdomen.items():
    for e in v:
        image_list_val_Abdomen.append((k,e))
random.shuffle(image_list_val_Abdomen)

train_loader_Abdomen = torch.utils.data.DataLoader(ImageFolder2(transform_train_Abdomen, image_list_train_Abdomen), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
val_loader_Abdomen = torch.utils.data.DataLoader(ImageFolder2(transform_test_Abdomen, image_list_val_Abdomen), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)


class_indexes_train_Bone = {}
for folder in folders:
    for file in os.listdir(os.path.join(train_dir_Bone, folder)):
        if folders.index(folder) in class_indexes_train_Bone:
            class_indexes_train_Bone[folders.index(folder)].append(os.path.join(train_dir_Bone, folder, file))
        else:
            class_indexes_train_Bone[folders.index(folder)] = [os.path.join(train_dir_Bone, folder, file)]

for k, v in class_indexes_train_Bone.items():
    v.sort()

class_indexes_val_Bone = {}
for folder in folders:
    for file in os.listdir(os.path.join(val_dir_Bone, folder)):
        if folders.index(folder) in class_indexes_val_Bone:
            class_indexes_val_Bone[folders.index(folder)].append(os.path.join(val_dir_Bone, folder, file))
        else:
            class_indexes_val_Bone[folders.index(folder)] = [os.path.join(val_dir_Bone, folder, file)]

for k, v in class_indexes_val_Bone.items():
    v.sort()


random.seed(args.seed)
for i in range(0, nb_classes):
    random.shuffle(class_indexes_train_Bone[i])

image_list_train_Bone = []
for k,v in class_indexes_train_Bone.items():
    for e in v:
        image_list_train_Bone.append((k,e))
random.shuffle(image_list_train_Bone)

for i in range(0, nb_classes):
    random.shuffle(class_indexes_val_Bone[i])

image_list_val_Bone = []
for k,v in class_indexes_val_Bone.items():
    for e in v:
        image_list_val_Bone.append((k,e))
random.shuffle(image_list_val_Bone)

train_loader_Bone = torch.utils.data.DataLoader(ImageFolder2(transform_train_Bone, image_list_train_Bone), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
val_loader_Bone = torch.utils.data.DataLoader(ImageFolder2(transform_test_Bone, image_list_val_Bone), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)


class_indexes_train_Spine = {}
for folder in folders:
    for file in os.listdir(os.path.join(train_dir_Spine, folder)):
        if folders.index(folder) in class_indexes_train_Spine:
            class_indexes_train_Spine[folders.index(folder)].append(os.path.join(train_dir_Spine, folder, file))
        else:
            class_indexes_train_Spine[folders.index(folder)] = [os.path.join(train_dir_Spine, folder, file)]

for k, v in class_indexes_train_Spine.items():
    v.sort()


class_indexes_val_Spine = {}
for folder in folders:
    for file in os.listdir(os.path.join(val_dir_Spine, folder)):
        if folders.index(folder) in class_indexes_val_Spine:
            class_indexes_val_Spine[folders.index(folder)].append(os.path.join(val_dir_Spine, folder, file))
        else:
            class_indexes_val_Spine[folders.index(folder)] = [os.path.join(val_dir_Spine, folder, file)]

for k, v in class_indexes_val_Spine.items():
    v.sort()


random.seed(args.seed)
for i in range(0, nb_classes):
    random.shuffle(class_indexes_train_Spine[i])

image_list_train_Spine = []
for k,v in class_indexes_train_Spine.items():
    for e in v:
        image_list_train_Spine.append((k,e))
random.shuffle(image_list_train_Spine)

for i in range(0, nb_classes):
    random.shuffle(class_indexes_val_Spine[i])

image_list_val_Spine = []
for k,v in class_indexes_val_Spine.items():
    for e in v:
        image_list_val_Spine.append((k,e))
random.shuffle(image_list_val_Spine)

train_loader_Spine = torch.utils.data.DataLoader(ImageFolder2(transform_train_Spine, image_list_train_Spine), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
val_loader_Spine = torch.utils.data.DataLoader(ImageFolder2(transform_test_Spine, image_list_val_Spine), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)


class_size = []
for folder in folders:
    class_size.append(len(class_indexes_train_Abdomen[folders.index(folder)])+len(class_indexes_val_Abdomen[folders.index(folder)])+len(class_indexes_train_Bone[folders.index(folder)])+len(class_indexes_val_Bone[folders.index(folder)])+len(class_indexes_train_Abdomen[folders.index(folder)])+len(class_indexes_val_Abdomen[folders.index(folder)]))

class_weights = [1-float(e)/sum(class_size) for e in class_size]
class_weights = torch.FloatTensor(class_weights).cuda()


learning_rate = args.lr

# Model
print('==> Building model ...')

model = APN(feat_size=args.fs, nb_prototypes=nb_classes, lamb=args.lamb, temp=args.temp, num_domains=3)
model.cuda()
print(model.parameters)


for name, m in model.named_parameters():
    if m.requires_grad:
        print(name)


criterion = nn.CrossEntropyLoss(weight=class_weights)

# learning rate
lr_plan = [learning_rate] * args.n_epoch


# Training
def train(epoch, lr_scheduler=None):
    print('\nEpoch: %d' % epoch)

    optimizer = torch.optim.SGD([
            {'params': model.predictor.parameters(), 'lr': lr_plan[epoch]},
            {'params': model.feat.parameters(), 'lr': lr_plan[epoch]/10.},
            ], weight_decay=0.0005)

    train_loss = 0
    correct_Abdomen = 0
    total_Abdomen = 0
    correct_Bone = 0
    total_Bone = 0
    correct_Spine = 0
    total_Spine = 0
    

    model.train()
    loader_train_Abdomen = iter(train_loader_Abdomen)
    loader_train_Bone = iter(train_loader_Bone)
    loader_train_Spine = iter(train_loader_Spine)
    iteration_Abdomen = len(train_loader_Abdomen)
    iteration_Bone = len(train_loader_Bone)
    iteration_Spine = len(train_loader_Spine)
    iteration = max(iteration_Abdomen, iteration_Bone, iteration_Spine)

    confusion_matrix_Abdomen = np.zeros((nb_classes, nb_classes))
    confusion_matrix_Bone = np.zeros((nb_classes, nb_classes))
    confusion_matrix_Spine = np.zeros((nb_classes, nb_classes))

    print('start training ...')
    for batch_idx in range(0, iteration):
        print('iterations: ', batch_idx)

        try:
            inputs_Abdomen, targets_Abdomen = next(loader_train_Abdomen)
        except:
            loader_train_Abdomen = iter(train_loader_Abdomen)
            inputs_Abdomen, targets_Abdomen = next(loader_train_Abdomen)
            
        inputs_Abdomen, targets_Abdomen = inputs_Abdomen.to(device), targets_Abdomen.to(device)
        
        try:
            inputs_Bone, targets_Bone = next(loader_train_Bone)
        except:
            loader_train_Bone = iter(train_loader_Bone)
            inputs_Bone, targets_Bone = next(loader_train_Bone)            

        inputs_Bone, targets_Bone = inputs_Bone.to(device), targets_Bone.to(device)

        try:
            inputs_Spine, targets_Spine = next(loader_train_Spine)
        except:
            loader_train_Spine = iter(train_loader_Spine)
            inputs_Spine, targets_Spine = next(loader_train_Spine)            

        inputs_Spine, targets_Spine = inputs_Spine.to(device), targets_Spine.to(device)

        optimizer.zero_grad()

        logits_Abdomen, reg_Abdomen = model(inputs_Abdomen, targets_Abdomen, epoch, 0)
        ce_loss_Abdomen = criterion(logits_Abdomen, targets_Abdomen)

        logits_Bone, reg_Bone = model(inputs_Bone, targets_Bone, epoch, 1)
        ce_loss_Bone = criterion(logits_Bone, targets_Bone)

        logits_Spine, reg_Spine = model(inputs_Spine, targets_Spine, epoch, 2)
        ce_loss_Spine = criterion(logits_Spine, targets_Spine)

        reg = reg_Abdomen + reg_Bone + reg_Spine

        loss = ce_loss_Abdomen + reg_Abdomen + ce_loss_Bone + reg_Bone + ce_loss_Spine + reg_Spine

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted_Abdomen = logits_Abdomen.max(1)
        total_Abdomen += targets_Abdomen.size(0)
        correct_Abdomen += predicted_Abdomen.eq(targets_Abdomen).sum().item()

        targets_Abdomen = list(targets_Abdomen.detach().cpu().numpy())
        predicted_Abdomen = list(predicted_Abdomen.detach().cpu().numpy())

        for i in range(0, len(targets_Abdomen)):
            confusion_matrix_Abdomen[targets_Abdomen[i]][predicted_Abdomen[i]] += 1


        _, predicted_Bone = logits_Bone.max(1)
        total_Bone += targets_Bone.size(0)
        correct_Bone += predicted_Bone.eq(targets_Bone).sum().item()

        targets_Bone = list(targets_Bone.detach().cpu().numpy())
        predicted_Bone = list(predicted_Bone.detach().cpu().numpy())

        for i in range(0, len(targets_Bone)):
            confusion_matrix_Bone[targets_Bone[i]][predicted_Bone[i]] += 1


        _, predicted_Spine = logits_Spine.max(1)
        total_Spine += targets_Spine.size(0)
        correct_Spine += predicted_Spine.eq(targets_Spine).sum().item()

        targets_Spine = list(targets_Spine.detach().cpu().numpy())
        predicted_Spine = list(predicted_Spine.detach().cpu().numpy())

        for i in range(0, len(targets_Spine)):
            confusion_matrix_Spine[targets_Spine[i]][predicted_Spine[i]] += 1

    accs_per_class_Abdomen = []
    for i in range(0, nb_classes):
        accs_per_class_Abdomen.append(confusion_matrix_Abdomen[i, i] / np.sum(confusion_matrix_Abdomen[i]))

    accs_per_class_Abdomen = np.array(accs_per_class_Abdomen)
    avg_acc_per_class_Abdomen = 100. * np.mean(accs_per_class_Abdomen)        

    train_acc_Abdomen = 100.*float(correct_Abdomen)/total_Abdomen

    for i in range(0, nb_classes):
        pps = ''
        for j in range(0, nb_classes):
            pps += str(confusion_matrix_Abdomen[i][j]) + ', '
        pps += str(round(100.*accs_per_class_Abdomen[i],2))
        print(pps)

    accs_per_class_Bone = []
    for i in range(0, nb_classes):
        accs_per_class_Bone.append(confusion_matrix_Bone[i, i] / np.sum(confusion_matrix_Bone[i]))

    accs_per_class_Bone = np.array(accs_per_class_Bone)
    avg_acc_per_class_Bone = 100. * np.mean(accs_per_class_Bone)        

    train_acc_Bone = 100.*float(correct_Bone)/total_Bone

    for i in range(0, nb_classes):
        pps = ''
        for j in range(0, nb_classes):
            pps += str(confusion_matrix_Bone[i][j]) + ', '
        pps += str(round(100.*accs_per_class_Bone[i],2))
        print(pps)

    accs_per_class_Spine = []
    for i in range(0, nb_classes):
        accs_per_class_Spine.append(confusion_matrix_Spine[i, i] / np.sum(confusion_matrix_Spine[i]))

    accs_per_class_Spine = np.array(accs_per_class_Spine)
    avg_acc_per_class_Spine = 100. * np.mean(accs_per_class_Spine)        

    train_acc_Spine = 100.*float(correct_Spine)/total_Spine

    for i in range(0, nb_classes):
        pps = ''
        for j in range(0, nb_classes):
            pps += str(confusion_matrix_Spine[i][j]) + ', '
        pps += str(round(100.*accs_per_class_Spine[i],2))
        print(pps)


    print ('Epoch [%d/%d], Lr: %F, Training Accuracy Abdomen: %.2F, Avg Acc Per Class Abdomen: %.2F, Training Accuracy Bone: %.2F, Avg Acc Per Class Bone: %.2F, Training Accuracy Spine: %.2F, Avg Acc Per Class Spine: %.2F, Loss: %.2f, reg: %.2f.' % (epoch+1, args.n_epoch, lr_plan[epoch], train_acc_Abdomen, avg_acc_per_class_Abdomen, train_acc_Bone, avg_acc_per_class_Bone, train_acc_Spine, avg_acc_per_class_Spine, loss.item(), reg.item()))

    return train_acc_Abdomen, avg_acc_per_class_Abdomen, train_acc_Bone, avg_acc_per_class_Bone, train_acc_Spine, avg_acc_per_class_Spine


def val(epoch):
    global best_acc_Abdomen
    global last_acc_Abdomen
    global best_avg_acc_Abdomen
    global last_avg_acc_Abdomen

    global best_acc_Bone
    global last_acc_Bone
    global best_avg_acc_Bone
    global last_avg_acc_Bone

    global best_acc_Spine
    global last_acc_Spine
    global best_avg_acc_Spine
    global last_avg_acc_Spine

    global image_list_test_Abdomen
    global image_list_test_Bone
    global image_list_test_Spine

    model.eval()

    correct_Abdomen = 0
    total_Abdomen = 0
    confusion_matrix_Abdomen = np.zeros((nb_classes, nb_classes))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader_Abdomen):
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, _ = model(inputs, None, epoch, 0)

            _, predicted = logits.max(1)
            total_Abdomen += targets.size(0)
            correct_Abdomen += predicted.eq(targets).sum().item()

            targets = list(targets.detach().cpu().numpy())
            predicted = list(predicted.detach().cpu().numpy())

            for i in range(0, len(targets)):
                confusion_matrix_Abdomen[targets[i]][predicted[i]] += 1

                
    accs_per_class_Abdomen = []
    for i in range(0, nb_classes):
        accs_per_class_Abdomen.append(confusion_matrix_Abdomen[i, i] / np.sum(confusion_matrix_Abdomen[i]))

    accs_per_class_Abdomen = np.array(accs_per_class_Abdomen)
    avg_acc_per_class_Abdomen = 100. * np.mean(accs_per_class_Abdomen)

    last_avg_acc_Abdomen = avg_acc_per_class_Abdomen


    for i in range(0, nb_classes):
        pps = ''
        for j in range(0, nb_classes):
            pps += str(confusion_matrix_Abdomen[i][j]) + '    '
        pps += str(round(100.*accs_per_class_Abdomen[i],2))
        print(pps)

    print('acc   ', 100.*(np.trace(confusion_matrix_Abdomen)/np.sum(confusion_matrix_Abdomen)), '  ', avg_acc_per_class_Abdomen)

    if avg_acc_per_class_Abdomen > best_avg_acc_Abdomen:
        best_avg_acc_Abdomen = avg_acc_per_class_Abdomen

    acc_Abdomen = 100.*correct_Abdomen/total_Abdomen
    last_acc_Abdomen = acc_Abdomen
    
    if acc_Abdomen > best_acc_Abdomen:
        best_acc_Abdomen = acc_Abdomen


    correct_Bone = 0
    total_Bone = 0
    confusion_matrix_Bone = np.zeros((nb_classes, nb_classes))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader_Bone):
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, _ = model(inputs, None, epoch, 1)

            _, predicted = logits.max(1)
            total_Bone += targets.size(0)
            correct_Bone += predicted.eq(targets).sum().item()

            targets = list(targets.detach().cpu().numpy())
            predicted = list(predicted.detach().cpu().numpy())

            for i in range(0, len(targets)):
                confusion_matrix_Bone[targets[i]][predicted[i]] += 1

    accs_per_class_Bone = []
    for i in range(0, nb_classes):
        accs_per_class_Bone.append(confusion_matrix_Bone[i, i] / np.sum(confusion_matrix_Bone[i]))

    accs_per_class_Bone = np.array(accs_per_class_Bone)
    avg_acc_per_class_Bone = 100. * np.mean(accs_per_class_Bone)

    last_avg_acc_Bone = avg_acc_per_class_Bone


    for i in range(0, nb_classes):
        pps = ''
        for j in range(0, nb_classes):
            pps += str(confusion_matrix_Bone[i][j]) + '    '
        pps += str(round(100.*accs_per_class_Bone[i],2))
        print(pps)

    print('acc   ', 100.*(np.trace(confusion_matrix_Bone)/np.sum(confusion_matrix_Bone)), '  ', avg_acc_per_class_Bone)

    if avg_acc_per_class_Bone > best_avg_acc_Bone:
        best_avg_acc_Bone = avg_acc_per_class_Bone

    acc_Bone = 100.*correct_Bone/total_Bone
    last_acc_Bone = acc_Bone
    
    if acc_Bone > best_acc_Bone:
        best_acc_Bone = acc_Bone



    correct_Spine = 0
    total_Spine = 0
    confusion_matrix_Spine = np.zeros((nb_classes, nb_classes))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader_Spine):
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, _ = model(inputs, None, epoch, 2)

            _, predicted = logits.max(1)
            total_Spine += targets.size(0)
            correct_Spine += predicted.eq(targets).sum().item()

            targets = list(targets.detach().cpu().numpy())
            predicted = list(predicted.detach().cpu().numpy())

            for i in range(0, len(targets)):
                confusion_matrix_Spine[targets[i]][predicted[i]] += 1

            
    accs_per_class_Spine = []
    for i in range(0, nb_classes):
        accs_per_class_Spine.append(confusion_matrix_Spine[i, i] / np.sum(confusion_matrix_Spine[i]))

    accs_per_class_Spine = np.array(accs_per_class_Spine)
    avg_acc_per_class_Spine = 100. * np.mean(accs_per_class_Spine)

    last_avg_acc_Spine = avg_acc_per_class_Spine


    for i in range(0, nb_classes):
        pps = ''
        for j in range(0, nb_classes):
            pps += str(confusion_matrix_Spine[i][j]) + '    '
        pps += str(round(100.*accs_per_class_Spine[i],2))
        print(pps)

    print('acc   ', 100.*(np.trace(confusion_matrix_Spine)/np.sum(confusion_matrix_Spine)), '  ', avg_acc_per_class_Spine)

    if avg_acc_per_class_Spine > best_avg_acc_Spine:
        best_avg_acc_Spine = avg_acc_per_class_Spine

    acc_Spine = 100.*correct_Spine/total_Spine
    last_acc_Spine = acc_Spine
    
    if acc_Spine > best_acc_Spine:
        best_acc_Spine = acc_Spine

    # Save checkpoint.
    if store_weights:
        print('Saving..')
        state1 = {
            'net': model.state_dict(),
            'acc_Abdomen': last_acc_Abdomen,
            'acc_Bone': last_acc_Bone,
            'acc_Spine': last_acc_Spine,
            'epoch': epoch,
        }
        if not os.path.isdir('/hdd8/zhulei/spine-mets/checkpoint-ct-bilsky-project'):
            os.mkdir('/hdd8/zhulei/spine-mets/checkpoint-ct-bilsky-project')

        torch.save(state1, '/hdd8/zhulei/spine-mets/checkpoint-ct-bilsky-project/windowing-aug-combine-dsbn-resnext-bilsky-three-model-'+str(args.kth)+'-'+str(args.lamb)+'-'+str(epoch)+'.t7')

    print ('Epoch [%d/%d], Acc Abdomen: %.2F, Best Acc Abdomen: %.2F, Avg Acc Per Class Abdomen: %.2F, Best Avg Acc Per Class Abdomen: %.2F, Acc Bone: %.2F, Best Acc Bone: %.2F, Avg Acc Per Class Bone: %.2F, Best Avg Acc Per Class Bone: %.2F, Acc Spine: %.2F, Best Acc Spine: %.2F, Avg Acc Per Class Spine: %.2F, Best Avg Acc Per Class Spine: %.2F.' % (epoch+1, args.n_epoch, acc_Abdomen, best_acc_Abdomen, avg_acc_per_class_Abdomen, best_avg_acc_Abdomen, acc_Bone, best_acc_Bone, avg_acc_per_class_Bone, best_avg_acc_Bone, acc_Spine, best_acc_Spine, avg_acc_per_class_Spine, best_avg_acc_Spine))
    
    return acc_Abdomen, acc_Bone, acc_Spine

# code for train
for epoch in range(start_epoch, args.n_epoch):
    train_acc_Abdomen, train_avg_acc_Abdomen, train_acc_Bone, train_avg_acc_Bone, train_acc_Spine, train_avg_acc_Spine = train(epoch)
    test_acc_Abdomen, test_acc_Bone, test_acc_Spine = val(epoch)

    with open('./records/windowing-combine-dsbn-resnext-bilsky-record-sgd-'+str(args.lr)+'-'+str(args.kth)+'-'+str(args.lamb)+'-'+str(args.temp)+'-'+str(args.n_epoch)+'-'+args.note+'.txt', "a") as myfile:
        myfile.write(str(args.kth) +'-'+ str(int(epoch)) + '-' + str(batch_size) + ': '  + str(train_acc_Abdomen) + ' ' + str(train_avg_acc_Abdomen) + ' ' + str(test_acc_Abdomen) + ' ' + str(best_acc_Abdomen) + ' ' + str(last_avg_acc_Abdomen) + ' ' + str(best_avg_acc_Abdomen) + ' ' + str(train_acc_Bone) + ' ' + str(train_avg_acc_Bone) + ' ' + str(test_acc_Bone) + ' ' + str(best_acc_Bone) + ' ' + str(last_avg_acc_Bone) + ' ' + str(best_avg_acc_Bone) + ' ' + str(train_acc_Spine) + ' ' + str(train_avg_acc_Spine) + ' ' + str(test_acc_Spine) + ' ' + str(best_acc_Spine) + ' ' + str(last_avg_acc_Spine) + ' ' + str(best_avg_acc_Spine) + "\n")

with open('./record-all.txt', 'a') as f:
    f.write('windowing-combine-dsbn-resnext-bilsky-'+str(args.kth)+'-sgd-'+str(args.lr)+'-'+args.note+'-'+str(args.lamb)+'-'+str(args.temp)+'-'+str(last_acc_Abdomen)+'-'+str(best_acc_Abdomen)+'-'+str(last_avg_acc_Abdomen)+'-'+str(best_avg_acc_Abdomen)+'-'+str(last_acc_Bone)+'-'+str(best_acc_Bone)+'-'+str(last_avg_acc_Bone)+'-'+str(best_avg_acc_Bone)+'-'+str(last_acc_Spine)+'-'+str(best_acc_Spine)+'-'+str(last_avg_acc_Spine)+'-'+str(best_avg_acc_Spine)+'\n')
