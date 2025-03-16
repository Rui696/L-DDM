import torch
import h5py
from timeit import default_timer

from Model.tools import MatRead, MaxMinNormalizer


def load_exp_data(data_folder, resolution, Nx, Ny, boundary_type):
    '''
    Load experimental data from .mat file
    Args:
        resolution: resolution of the data
        Nx: number of grids in x direction
        Ny: number of grids in y direction
        boundary_type: type of boundary condition
    Returns:
        a_field: [nsamples, 1, ygrids, xgrids]
        u_truth: [nsamples, ygrids, xgrids]
        ux_truth: [nsamples, ygrids-4, xgrids-4]
    '''
    t1 = default_timer()
    data_path = '{}/Exp_Data_{}_{}by{}_Rectangle_{}.mat'.format(data_folder, resolution, Nx, Ny, boundary_type)
    data = h5py.File(data_path)
    a_field = torch.tensor(data['a_field'][:], dtype=torch.float32).permute(2, 1, 0).unsqueeze(1)
    u_truth = torch.tensor(data['u_field'][:], dtype=torch.float32).permute(2, 1, 0)
    ux_truth = torch.tensor(data['ux_field'][:], dtype=torch.float32).permute(2, 1, 0)[:, 2:-2, 2:-2]
    t2 = default_timer()
    print('--Exp data loaded, time: {:.4f}'.format(t2-t1))

    return a_field, u_truth, ux_truth


def load_normalizer(data_folder, resolution):
    '''
    Parametrize the normalizer
    Args:
        data_folder: folder of the data
        resolution: resolution of the data
    Returns:
        a_normalizer: normalizer for a
        u_normalizer: normalizer for u
        ux_normalizer: normalizer for ux
    '''
    t1 = default_timer()
    if resolution == 65:
        train_path = '{}/Data_65_Dd_30kto40k.mat'.format(data_folder)
    elif resolution == 129:
        train_path = '{}/Data_129_Dd_20kto30k.mat'.format(data_folder)
    data = MatRead(train_path)
    train_a, train_u, train_ux, _ = data.get_data()

    a_normalizer = MaxMinNormalizer(train_a)
    u_normalizer = MaxMinNormalizer(train_u)
    ux_normalizer = MaxMinNormalizer(train_ux[:,:,2:-2,2:-2])

    t2 = default_timer()
    print('--Normalizer loaded, time: {:.4f}'.format(t2-t1))

    return a_normalizer, u_normalizer, ux_normalizer


def load_model(model, ckpt_dir, model_id, Nep, device):
    '''
    Load model from the model_path
    Args:
        model: model to be loaded
        ckpt_dir: directory of the checkpoint
        model_id: model id
        Nep: number of epochs when the model was saved
        device: device to load the model
    Returns:
        model: loaded model
    '''
    t1 = default_timer()
    resume_path = '{}/{}_checkpoint_{}.pth'.format(ckpt_dir, model_id, Nep-1)
    checkpoint = torch.load(resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['net'])
    model.to(device)

    t2 = default_timer()
    print('--Model loaded from: {}'.format(resume_path))
    print('               time: {:.4f}'.format(t2-t1))

    return model


def init_a(a_field, Nx, Ny, nsamples, resolution, nonoverlap_nodes, a_normalizer, device):
    '''
    Initialize a_subdomain [Ny, Nx, nsamples, 1, resolution, resolution]
    '''
    a_subdomain = torch.empty((Ny, Nx, nsamples, 1, resolution, resolution))
    for j in range(Ny):
        idy = j * nonoverlap_nodes
        for i in range(Nx):
            idx = i * nonoverlap_nodes
            a_subdomain[j, i] = a_field[:, :, idy:(idy+resolution), idx:(idx+resolution)]
    a_subdomain = a_normalizer.encode(a_subdomain).to(device)
    return a_subdomain


def init_u(bc_truth, xgrids, ygrids):
    '''
    Initialize u field
    '''
    y, x = torch.meshgrid(torch.linspace(0,1,ygrids), torch.linspace(0,1,xgrids), indexing='ij')
    points = torch.stack([x, y], dim=2).flatten(0,1)

    b_T = torch.stack([torch.linspace(0,1,xgrids), torch.ones(xgrids)], dim=1)
    b_R = torch.stack([torch.ones(ygrids), torch.linspace(0,1,ygrids)], dim=1)
    b_B = torch.stack([torch.linspace(0,1,xgrids), torch.zeros(xgrids)], dim=1)
    b_L = torch.stack([torch.zeros(ygrids), torch.linspace(0,1,ygrids)], dim=1)
    boundary = torch.cat([b_T[:-1], b_R[1:], b_B[1:], b_L[:-1]], dim=0)

    distances = torch.cdist(points, boundary).reshape(ygrids, xgrids, -1)

    f_distance = 1 / (distances)**1
    f_distance = torch.div(f_distance, torch.sum(f_distance, dim=2).unsqueeze(2)) # *(N-1)

    initial = torch.matmul(f_distance, bc_truth.permute(1, 0)).permute(2, 0, 1) # /(N-1) # [nsamples, ygrids, xgrids]

    initial[:, -1, :-1] = bc_truth[:, :xgrids-1]
    initial[:, 1:, -1] = bc_truth[:, xgrids-1:xgrids+ygrids-2]
    initial[:, 0, 1:] = bc_truth[:, xgrids+ygrids-2:2*xgrids+ygrids-3]
    initial[:, :-1, 0] = bc_truth[:, 2*xgrids+ygrids-3:]

    return initial

def partition_of_unity(nonoverlap_nodes, overlap_nodes, resolution, nsamples, device):
    '''
    Partition of unity for the initial condition
    '''
    ki_T_u = torch.cat([torch.ones(nonoverlap_nodes, resolution), torch.linspace(1,0,overlap_nodes).unsqueeze(1).expand(overlap_nodes,resolution)], 
                      dim=0).unsqueeze(0).expand(nsamples,resolution,resolution).to(device)
    ki_R_u = torch.cat([torch.ones(resolution, nonoverlap_nodes), torch.linspace(1,0,overlap_nodes).unsqueeze(0).expand(resolution,overlap_nodes)], 
                      dim=1).unsqueeze(0).expand(nsamples,resolution,resolution).to(device)
    ki_B_u = torch.cat([torch.linspace(0,1,overlap_nodes).unsqueeze(1).expand(overlap_nodes,resolution), torch.ones(nonoverlap_nodes, resolution)], 
                      dim=0).unsqueeze(0).expand(nsamples,resolution,resolution).to(device)
    ki_L_u = torch.cat([torch.linspace(0,1,overlap_nodes).unsqueeze(0).expand(resolution,overlap_nodes), torch.ones(resolution, nonoverlap_nodes)], 
                      dim=1).unsqueeze(0).expand(nsamples,resolution,resolution).to(device)

    ki_T_ux = torch.cat([torch.ones(nonoverlap_nodes, resolution-4), torch.linspace(1,0,overlap_nodes-4).unsqueeze(1).expand(overlap_nodes-4,resolution-4)], 
                       dim=0).unsqueeze(0).expand(nsamples,resolution-4,resolution-4).to(device)
    ki_R_ux = torch.cat([torch.ones(resolution-4, nonoverlap_nodes), torch.linspace(1,0,overlap_nodes-4).unsqueeze(0).expand(resolution-4,overlap_nodes-4)], 
                       dim=1).unsqueeze(0).expand(nsamples,resolution-4,resolution-4).to(device)
    ki_B_ux = torch.cat([torch.linspace(0,1,overlap_nodes-4).unsqueeze(1).expand(overlap_nodes-4,resolution-4), torch.ones(nonoverlap_nodes, resolution-4)], 
                       dim=0).unsqueeze(0).expand(nsamples,resolution-4,resolution-4).to(device)
    ki_L_ux = torch.cat([torch.linspace(0,1,overlap_nodes-4).unsqueeze(0).expand(resolution-4,overlap_nodes-4), torch.ones(resolution-4, nonoverlap_nodes)], 
                      dim=1).unsqueeze(0).expand(nsamples,resolution-4,resolution-4).to(device)
    
    return ki_T_u, ki_R_u, ki_B_u, ki_L_u, ki_T_ux, ki_R_ux, ki_B_ux, ki_L_ux