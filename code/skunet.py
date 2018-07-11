# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

class SKUNET(nn.Module):
    def __init__(self):
        super(SKUNET, self).__init__()

        self.relu = nn.ReLU()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.in_layer = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.in_layer_bn = nn.BatchNorm3d(32)

        self.lconvlayer1 = nn.Conv3d(32, 32, kernel_size=3, stride=(1, 2, 2), padding=1)  # 128, 56

        self.lconvlayer2 = nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1)  # 64, 56
        self.lconvlayer2_bn = nn.BatchNorm3d(64)

        self.lconvlayer3 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)  # 32, 23

        self.lconvlayer4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)  # 16,
        self.lconvlayer4_bn = nn.BatchNorm3d(128)

        self.lconvlayer5 = nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1)  # 10

        self.lconvlayer6 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)  # 5
        self.lconvlayer6_bn = nn.BatchNorm3d(256)

        self.rconvTlayer6 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.rconvlayer6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.rconvTlayer5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer5_bn = nn.BatchNorm3d(128)

        self.rconvTlayer4 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.rconvlayer4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.rconvTlayer3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.rconvlayer3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.rconvlayer3_bn = nn.BatchNorm3d(64)

        self.rconvTlayer2 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.rconvlayer2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.rconvTlayer1 = nn.ConvTranspose3d(32, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.rconvlayer1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.rconvlayer1_bn = nn.BatchNorm3d(32)

        self.out_layer = nn.Conv3d(32, 1, kernel_size=1, stride=1)

    def forward(self, x):
        xl0 = self.relu(self.in_layer_bn(self.in_layer(x)))
        #        print('xl0')
        #        print(xl0.shape)
        xl1 = self.relu(self.lconvlayer1(xl0))
        #        print('xl1')
        #        print(xl1.shape)
        xl2 = self.relu(self.lconvlayer2_bn(self.lconvlayer2(xl1)))
        #        print('xl2')
        #        print(xl2.shape)
        xl3 = self.relu(self.lconvlayer3(xl2))
        #        print('xl3')
        #        print(xl3.shape)
        xl4 = self.relu(self.lconvlayer4_bn(self.lconvlayer4(xl3)))
        #        print('xl4')
        #        print(xl4.shape)
        xl5 = self.relu(self.lconvlayer5(xl4))
        #        print('xl5')
        #        print(xl5.shape)
        xl6 = self.relu(self.lconvlayer6_bn(self.lconvlayer6(xl5)))
        #        print('xl6')
        #        print(xl6.shape)

        ###        xl7 = self.relu(self.lconvlayer7(xl6))

        ###        xr7 = xl7

        ###        xr7 = self.relu(self.rconvTlayer7(xr7))

        ###        print('xr7')
        ###        print(xr7.shape)

        ###        xr6 = torch.add(xr7, xl6)

        ###        print('xr6')
        ###        print(xr6.shape)
        ###        xr6 = self.relu(self.rconvTlayer6(self.relu(self.rconvlayer7_bn(self.rconvlayer7(xr6)))))
        ###        print('xr6')
        ###        print(xr6.shape)

        xr6 = xl6

        xr6 = self.relu(self.rconvTlayer6(self.relu(self.upsample(xr6))))

        xr5 = torch.add(xr6, xl5)
        #        print('xr5')
        #        print(xr5.shape)
        xr5 = self.relu(self.rconvTlayer5(self.relu(self.upsample(self.relu(self.rconvlayer6(xr5))))))

        #        print('dual_shape', xr5.shape, xl4.shape)
        xr4 = torch.add(xr5, xl4)
        #        print('xr4')
        #        print(xr4.shape)
        xr4 = self.relu(
            self.rconvTlayer4(self.relu(self.upsample(self.relu(self.rconvlayer5_bn(self.rconvlayer5(xr4)))))))

        xr3 = torch.add(xr4, xl3)
        #        print('xr3')
        #        print(xr3.shape)
        xr3 = self.relu(self.rconvTlayer3(self.relu(self.upsample(self.relu(self.rconvlayer4(xr3))))))

        xr2 = torch.add(xr3, xl2)
        #        print('xr2')
        #        print(xr2.shape)
        xr2 = self.relu(self.rconvTlayer2(self.relu(self.rconvlayer3_bn(self.rconvlayer3(xr2)))))

        xr1 = torch.add(xr2, xl1)
        #        print('xr1')
        #        print(xr1.shape)
        xr1 = self.relu(self.rconvTlayer1(self.relu(self.rconvlayer2(xr1))))

        xr0 = torch.add(xr1, xl0)

        xr0 = self.relu(self.rconvlayer1_bn(self.rconvlayer1(xr0)))

        #        print('xr0')
        #        print(xr0.shape)

        out_layer = self.out_layer(xr0)

        return out_layer