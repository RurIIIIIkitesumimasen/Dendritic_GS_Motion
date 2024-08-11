from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0')



class OrientationNet(nn.Module):
    def __init__(
        self,
        dendrite=1,
        init_w_mul=0.01,
        init_w_add=0.2,
        init_q=0,
        pad=1,
        k=10,
    ):
        super(OrientationNet, self).__init__()
        self.frontconv = FrontConv(
            pad=pad
        )
        # self.dconvOnOff = FrontConvOnOffResponse(
        #     pad=pad
        # )
        self.dconvSynaps = DConvSynaps(
            dendrite=dendrite,
            init_w_mul=init_w_mul,
            init_w_add=init_w_add,
            init_q=init_q,
            k=k
        )
        self.dconvDend = DConvDend()
        self.dconvMenb = DConvMenb()
        self.dconvSoma = DConvSoma()
        self.calcOutput = CalcOutput()

    # @profile
    def forward(self, x):
        x = self.frontconv(x)
        # x = self.dconvOnOff(x)
        x = self.dconvSynaps(x)
        x = self.dconvDend(x)
        x = self.dconvMenb(x)
        x = self.dconvSoma(x)
        x = self.calcOutput(x)

        return x


class FrontConv(nn.Module):
    def __init__(
        self,
        input_dim=((2, 128, 128)),
        output_dim=((16384, 9)),
        filter_size=3,
        pad=1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filter_size = filter_size
        self.pad = pad
        image_w = input_dim[2]
        self.activate = nn.Sigmoid()

    def forward(self, x):
        # im2col
        x = nn.Unfold(kernel_size=(self.filter_size, self.filter_size), stride=(
            1, 1), padding=0, dilation=(1, 1))(x)

        x_before = x[:, :9, :]
        x_after = x[:, 9:18, :]


# 中心と周囲の入力の取り出し
#on off response
        x_center = torch.cat(
        [x_before[:, 4, ...].unsqueeze(0)for _ in range(9)], dim=0)
        x_center = x_center.permute(1, 0, 2)
        #x_center = x_center.permute(1,0,2,3)

# 中心と周囲の比較
        x = torch.isclose(x_after, x_center, rtol=0, atol=0)
        # x[:, 4, ...] = ~x[:, 4, ...]
        x = x.float()
        # x[:, 4, :] = torch.logical_not(torch.isclose(
        # x[:, 4, :], x[:, 4, :], rtol=0, atol=self.err_center))
        # #x[:,4,:] = (x[:,4,:,0] != x[:,4,:,0]) | (x[:,4,:,1] != x[:,4,:,1]) | (x[:,4,:,2] != x[:,4,:,2])
        # #x[:,4,:] = tmp[...,0] | tmp[...,1] | tmp[...,2]
       # print(x.shape)
        x = x.permute(0, 2, 1)
        return x





class DConvSynaps(nn.Module):
    def __init__(
        self,
        dendrite=1,
        init_w_mul=0.01,
        init_w_add=0.2,
        init_q=0,
        k=10
    ):
        self.dendrite = dendrite
        super().__init__()
        self.W = nn.Parameter(
            torch.Tensor(
                init_w_mul * 
                #np.ones((self.dendrite, 18, 4))
                #np.abs(
                np.random.randn(self.dendrite, 9, 8)
                #)
                +init_w_add #
            ))
        self.q = nn.Parameter(
            torch.Tensor(
                init_w_mul * 
                #np.ones((self.dendrite, 18, 4))
                #np.abs(
                np.random.randn(self.dendrite, 9, 8)
                #)
                +init_q #.random.randn
            ))
        self.activation = nn.Sigmoid()

        self.k = k

    def forward(self, x):
        x_width = x.shape[1]
        W = self.W.expand(x.shape[0], x_width, self.dendrite,
                          9, 8)
        q = self.q.expand(x.shape[0], x_width, self.dendrite,
                          9, 8)
        x = torch.cat([x.unsqueeze(0) for _ in range(8)], dim=0)
        x = x.unsqueeze(0)
        x = x.permute(2, 3, 0, 4, 1)
        return self.activation((x.to(device) * W - q) * self.k)


class DConvDend(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return torch.prod(x, 3)


class DConvMenb(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, 2)


class DConvSoma(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation((x - 0.5) * 10)


class CalcOutput(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, 1)
