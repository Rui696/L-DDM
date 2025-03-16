import torch

import numpy as np

# from functools import cache
import functools

def lagrange_polynomial(x, nodes, index):
    m = nodes.shape[0]
    x = torch.repeat_interleave(x.unsqueeze(1), m-1, 1)
    ind = np.r_[0:index,(index+1):m]

    y = torch.prod((x - nodes[ind]) / (nodes[index] - nodes[ind]), dim=1)

    return y

def lagrange_polynomials(x, nodes):
    return torch.stack([lagrange_polynomial(x, nodes, k) for k in range(nodes.shape[0])]).T

# @cache
@functools.lru_cache(None)
def lagrange_interp_mat(old_kernel, new_kernel, device):
    nodes = torch.linspace(-1., 1., old_kernel)
    new_nodes = torch.linspace(-1., 1., new_kernel)

    interp_mat = lagrange_polynomials(new_nodes, nodes)
    interp_mat = torch.kron(interp_mat, interp_mat)

    return interp_mat.to(device)

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Conv2d(torch.nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel,
                 bias=True,
                 downsample=False,
                 init_mode='kaiming_normal', 
                 init_weight=1, 
                 init_bias=0
                ):
        super().__init__()
        assert kernel % 2 != 0 and kernel >= 3

        self.intep_modes = ['lagrange', 'nearest-exact', 'bilinear', 'bicubic']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel

        if downsample:
            self.stride = 2
        else:
            self.stride = 1

        init_kwargs = dict(mode=init_mode, 
                           fan_in=in_channels*kernel*kernel, 
                           fan_out=out_channels*kernel*kernel)

        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, 
                                                      kernel, kernel], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if bias else None
    
    def get_weight(self, level, interp_mode):
        new_kernel = int((self.kernel - 1)*(2**level) + 1)

        if level > 0:
            antialias = True if interp_mode == 'bilinear' or interp_mode == 'bicubic' else False
            weight = torch.nn.functional.interpolate(
                self.weight, 
                size=(new_kernel,new_kernel), 
                mode=interp_mode,
            )
        elif level < 0:
            s = int(2**(-level))
            weight = self.weight[...,::s,::s] 
        else:
            weight = self.weight
        
        return weight
    
    def get_padding(self, level):
        new_kernel = int((self.kernel - 1)*(2**level) + 1)
        return (new_kernel - 1) // 2
    
    def get_scale(self, level):
        new_kernel = int((self.kernel - 1)*(2**level) + 1)
        return float((self.kernel)**2) / (new_kernel)**2 ## changed

    def forward(self, x, level=0, interp_mode='bilinear', pad_mode='constant'):
        assert interp_mode in self.intep_modes

        weight = self.get_weight(level, interp_mode)
        weight = self.get_scale(level)*weight

        pad = self.get_padding(level)
        if pad > 0:
            x = torch.nn.functional.pad(x, (pad,)*4, mode=pad_mode)

        x = torch.nn.functional.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=0)

        return x