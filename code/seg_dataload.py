# -*- coding: UTF-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import time
import cv2
import nrrd



class Data_set_seg(Dataset):


    def __init__(self, root_dir):
        self.root_dir = os.path.expanduser(root_dir)
        # 存储每个病人的文件信息
        self.data_list = []

        self.item_length = []

        self.data_length = 0


        #用于getitem
        # 当前已读取病人的索引号
        self.current_patient_index = -1
        patients = os.listdir(root_dir)


        # 记录相关信息
        self.data_length = len(patients)
        self.patients = patients
        print(patients)
        print('Data num: %d' % self.data_length)


    def __len__(self):
        return self.data_length


    def __nrrd2np__(self, Nrrd):
        nrrd_data, nrrd_option = nrrd.read(Nrrd)
        nrrd_array = np.array(nrrd_data,dtype=float)


        # 这里进行矩阵的归一化 可以尝试将这一步省略
        nrrd_min, nrrd_max = nrrd_array.min(), nrrd_array.max()
        nrrd_array = (nrrd_array- nrrd_min)/(nrrd_max - nrrd_min)

        # 改变矩阵的维度
        resize = np.zeros((512,512,64), dtype=float)
        output_array = np.zeros((128,128,64), dtype=float)
        for i in range(min(64, nrrd_array.shape[2])):
            resize[:, :, i] = nrrd_array[:, :, i]
        for i in range(resize.shape[2]):
            output_array[:,:,i] = cv2.resize(resize[:,:,i], (128, 128),interpolation=cv2.INTER_CUBIC )
        return output_array

    def __getitem__(self, idx):
        '''
        返回（相对应坐标，tag）
        :param idx: ？？？？
        :return:
        '''

        patient = self.patients[idx]

        input_patient_file = self.__nrrd2np__(self.root_dir + '/' + patient + '/' + patient + '+' + '.nrrd')
        label_tag_img = self.__nrrd2np__(self.root_dir + '/' + patient + '/' + patient + '+_label' + '.nrrd')


        return torch.from_numpy(np.array([input_patient_file.swapaxes(0,2)])).float(), \
               torch.from_numpy(np.array([label_tag_img.swapaxes(0,2)])).float()



