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
from seg_dataload import Data_set_seg
from torch.autograd import Variable
from skunet import SKUNET
import numpy as np
import copy
import matplotlib.pyplot as plt
import nrrd

from utils import progress_bar

if __name__ == '__main__':

    scale = 256

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", help="input folder path")  # 存放测试数据
    parser.add_argument("--cpu", help="use cpu", default='False')

    args = parser.parse_args()
    test_dir = args.test_dir
    cpu = json.loads(args.cpu.lower())

    net = SKUNET()
    model = torch.load('./output256_1000/256-900.pkl', map_location=lambda storage, loc: storage)
    new_model = {}
    for key, value in model.items():
        new_key = key[7:]
        new_model[new_key] = value
    net.load_state_dict(new_model)
    '''
    use_cuda = torch.cuda.is_available()
    print('Use cuda: ' + str(use_cuda))



    if use_cuda and not cpu:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net.cuda()
        cudnn.benchmark = True
    '''
    testset = Data_set_seg('./test_data')

    trainloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    datanum = 14

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        datanum += 1
        output_seg_results = []

        inputs = Variable(inputs)

        outputs = net(inputs)

        outputs_sig = nn.Sigmoid()(outputs)

        outputs_labes = outputs_sig.data

        for i in range(len(outputs_labes)):

            outputs_labe = outputs_labes[i][0]
            outputs_labe[outputs_labe >= 0.5] = 1
            outputs_labe[outputs_labe < 0.5] = 0

            output_seg_results.append((copy.deepcopy(np.array(outputs_labe))))
            output_seg_results = np.array(output_seg_results).swapaxes(1, 3)

            print(output_seg_results.shape)


        #
        #     np.save('./output/out'+str(i)+'.npy', output_seg_results)
        #


        clean = Data_set_seg.__nrrd2np__('./test_data', './test_data/' + str(datanum) + '/' + str(datanum) + '+.nrrd')
        label = Data_set_seg.__nrrd2np__('./test_data', './test_data/' + str(datanum) + '/' + str(datanum) + '+_label.nrrd')

        nrrd.write('./result/' + str(datanum)+'/' + str(datanum) + '+label.nrrd', output_seg_results[0])
        nrrd.write('./result/' + str(datanum)+'/' + str(datanum) + '+.nrrd', clean)
        nrrd.write('./result/' + str(datanum)+'/' + str(datanum) + '+true_label.nrrd', label)

        # for i in range(64):
        #     plt.imsave('./result/256/15/result_%s.png' % i, output_seg_results[0, :, :, i])

        # for i in range(64):
        #     n = 0
        #     for j in range(scale):
        #         for k in range(scale):
        #             if output_seg_results[0, j, k, i] != 0:
        #                 clean[j, k, i] = output_seg_results[0, j, k, i]

        # nrrd.write('./result_nrrd/'+str(datanum)+'+.nrrd', clean)

            # plt.imsave('./result_merge/256/15/%s.png' % i, clean[:, :, i])



