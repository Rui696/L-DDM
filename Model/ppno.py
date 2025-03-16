import torch
import torch.nn as nn
from .unet_2d import UNet2DModel


class UNet(nn.Module):
    def __init__(
            self,
            sample_size: int = 129,
            in_channels: int = 2,
            out_channels: int = 1,
            kernel_size: int = 3,
        ):
        '''
        Construct PPNO model based on UNet2DModel
        '''
        super(UNet, self).__init__()
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            down_block_types=('DownBlock2D', 'DownBlock2D', 'DownBlock2D'),
            up_block_types=('UpBlock2D', 'UpBlock2D', 'UpBlock2D'),
            block_out_channels=(128, 256, 384),
            layers_per_block=2,
            mid_block_scale_factor=1,
            downsample_type='conv',
            upsample_type='conv',
            dropout=0.0,
            act_fn='silu',
            attention_head_dim=8,
            norm_num_groups=32,
            attn_norm_num_groups=None,
            norm_eps=1e-05,
            add_attention=False,
        )
        self.f_distances = {}

    def forward(self, a, bc, level=0, interp_mode='bilinear', pad_mode='constant'):
        if a.shape[-1] not in self.f_distances:
            self.generate_distance_function(a.shape[-1], a.device)
        boundary = self.boundary_processor(bc, a.shape[-1])
        x = torch.cat((a, boundary), dim=1)

        return self.unet(x, level=level, interp_mode=interp_mode, pad_mode=pad_mode)

    def generate_distance_function(self, res, device):
        '''
        Generate the kenerl to interpolate over the boundaries (based on resolution)
        '''
        y, x = torch.meshgrid(torch.linspace(0,1,res), torch.linspace(0,1,res), indexing='ij')
        points = torch.stack([x, y], dim=2).flatten(0,1)

        b_T = torch.stack([torch.linspace(0,1,res), torch.ones(res)], dim=1)
        b_R = torch.stack([torch.ones(res), torch.linspace(0,1,res)], dim=1)
        b_B = torch.stack([torch.linspace(0,1,res), torch.zeros(res)], dim=1)
        b_L = torch.stack([torch.zeros(res), torch.linspace(0,1,res)], dim=1)
        boundary = torch.cat([b_T[:-1], b_R[1:], b_B[1:], b_L[:-1]], dim=0)

        distances = torch.cdist(points, boundary).reshape(res, res, -1)

        f_distance = 1 / (distances)**2
        f_distance = torch.div(f_distance, torch.sum(f_distance, dim=2).unsqueeze(2)) # *(N-1)

        self.f_distances[res] = f_distance.to(device)
    
    def boundary_processor(self, bc, res):
        '''
        Process the boundary conditions to interpolate over the boundaries
        '''
        integral = torch.matmul(self.f_distances[res], bc.permute(1, 0)).permute(2, 0, 1) # /(N-1)

        # Assign the boundary values (based on how bc is concatenated)
        integral[:, -1, :-1] = bc[:, :res-1]
        integral[:, 1:, -1]  = bc[:, res-1:2*(res-1)]
        integral[:, 0, 1:]   = bc[:, 2*(res-1):3*(res-1)]
        integral[:, :-1, 0]  = bc[:, 3*(res-1):]

        return integral.unsqueeze(1)
