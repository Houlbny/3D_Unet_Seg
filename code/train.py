
# -*- coding: UTF-8 -*-
from __future__ import print_function
import json
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from seg_dataload import Dataset_seg
from skunet import SKUNET
from utils import progress_bar


DUMPING_CYCLE = 100


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    iflat = nn.Sigmoid()(iflat)
    tflat = target.view(-1)


    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def houl_loss(input, target, *scribbles):
    smooth = 1.

    iflat = input.view(-1)
    iflat = nn.Sigmoid()(iflat)
    tflat = target.view(-1)
    intersection1 = (iflat * tflat).sum()
    if scribbles:
        s = scribbles[0]
        sflat = s.view(-1)
        intersection2 = (iflat * sflat).sum()
        return 1 - 0.3 * ((2. * intersection1 + smooth) / (iflat.sum() + tflat.sum() + smooth)) - \
                   0.7 * ((2. * intersection2 + smooth) / (iflat.sum() + sflat.sum() + smooth))
    else:
        return 1 - ((2. * intersection1 + smooth) / (iflat.sum() + tflat.sum() + smooth))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="dicom folder path")
    parser.add_argument("--output_dir", help="output folder path", default='.')
    parser.add_argument("--load_model", help="load model param", default=None)
    parser.add_argument("--batch_size", help="batch size", default='32')
    parser.add_argument("--workers_num", help="num of data loader workers", default='4')
    parser.add_argument("--epoch", help="epoch", default='500')
    parser.add_argument("--out_param_file", help="output param file name", default='tmp-calclfication-model')
    parser.add_argument("--cpu", help="use cpu", default='False')

    args = parser.parse_args()

    # 输入数据目录
    input_dir = args.input_dir

    if not input_dir:
        print('Must give input folder path as 1st param')
        exit(1)

    out_dir = args.output_dir
    load_model = args.load_model

    batch_size = int(args.batch_size)
    num_workers = int(args.workers_num)
    epoch_num = int(args.epoch)
    cpu = json.loads(args.cpu.lower())
    out_param_file = args.out_param_file


    use_cuda = torch.cuda.is_available()
    print('Use cuda: ' + str(use_cuda))


    # 载入训练数据


    trainset = Dataset_seg(input_dir)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    net = SKUNET()

    if use_cuda and not cpu:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net.cuda()
        cudnn.benchmark = True


    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=0.001)


    if load_model:
        if not cpu:
            net.load_state_dict(torch.load(load_model))
        else:
            net.load_state_dict(torch.load(load_model, map_location=lambda storage, loc: storage))


    def train(now_epoch):
        print('\nEpoch: %d' % now_epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        i = 0
        loss_sum = 0
        # train
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            
            if use_cuda and not cpu:
                inputs, targets= inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            # 输入整张切片
            inputs,  targets = Variable(inputs),  Variable(targets)

            outputs = net(inputs)

            loss = dice_loss(outputs, targets)
            print(loss)
            loss_sum += loss.data[0]
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f )'
                         % (train_loss / (batch_idx + 1)))
            i += 1
            if i % DUMPING_CYCLE == 0 or now_epoch % DUMPING_CYCLE == 0:
                out_param = out_param_file + "-" + str(now_epoch) + "-" + time.strftime("%Y%m%d") + ".pkl"
                torch.save(net.state_dict(),
                           os.path.join(out_dir,
                                        out_param))



    for epoch in range(0, epoch_num):
        train(epoch)
