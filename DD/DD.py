import torch
import torch.utils.data as Data


def update_boundary_condition(Nx, Ny, nsamples, nonoverlap_nodes, resolution, u_prediction, device):
    '''
    Update boundary condition
    Returns:
        bc_eval [Ny*Nx*nsamples, 4*(resolution)]
    '''
    bc_subdomain = torch.empty((Ny, Nx, nsamples, (resolution-1)*4)).to(device)
    for j in range(Ny):
        idy = j*nonoverlap_nodes
        for i in range(Nx):
            idx = i*nonoverlap_nodes
            bc_subdomain[j, i, :, :resolution-1] = u_prediction[:, idy+resolution-1, idx:idx+resolution-1]
            bc_subdomain[j, i, :, resolution-1:2*(resolution-1)] = u_prediction[:, idy+1:idy+resolution, idx+resolution-1]
            bc_subdomain[j, i, :, 2*(resolution-1):3*(resolution-1)] = u_prediction[:, idy, idx+1:idx+resolution]
            bc_subdomain[j, i, :, 3*(resolution-1):] = u_prediction[:, idy:idy+resolution-1, idx]
    bc_eval = bc_subdomain.flatten(0,2)

    return bc_eval


def reassemble_prediction(nsamples, ygrids, xgrids, u_subdomain, ux_subdomain, resolution, nonoverlap_nodes,
                          Ny, Nx, ki_T_u, ki_R_u, ki_B_u, ki_L_u, ki_T_ux, ki_R_ux, ki_B_ux, ki_L_ux,
                          bc_truth_enc, predict_ux, device):
    '''
    Reassemble the prediction
    '''
    u_prediction = torch.zeros((nsamples, ygrids, xgrids)).to(device)
    ux_prediction = torch.zeros((nsamples, ygrids-4, xgrids-4)).to(device)

    u_subdomain[:-1, :] *= ki_T_u.view(1, 1, nsamples, resolution, resolution)  # Top row
    u_subdomain[:, :-1] *= ki_R_u.view(1, 1, nsamples, resolution, resolution)  # Right column
    u_subdomain[1:, :] *= ki_B_u.view(1, 1, nsamples, resolution, resolution)  # Bottom row
    u_subdomain[:, 1:] *= ki_L_u.view(1, 1, nsamples, resolution, resolution)  # Left column

    if predict_ux:
        ux_subdomain[:-1, :] *= ki_T_ux.view(1, 1, nsamples, resolution-4, resolution-4)  # Top row
        ux_subdomain[:, :-1] *= ki_R_ux.view(1, 1, nsamples, resolution-4, resolution-4)  # Right column
        ux_subdomain[1:, :] *= ki_B_ux.view(1, 1, nsamples, resolution-4, resolution-4)  # Bottom row
        ux_subdomain[:, 1:] *= ki_L_ux.view(1, 1, nsamples, resolution-4, resolution-4)  # Left column

    for j in range(Ny):
        idy = j * nonoverlap_nodes
        for i in range(Nx):
            idx = i * nonoverlap_nodes
            u_prediction[:, idy:(idy+resolution), idx:(idx+resolution)] += u_subdomain[j, i]
            if predict_ux:
                ux_prediction[:, idy:(idy+resolution-4), idx:(idx+resolution-4)] += ux_subdomain[j, i]

    u_prediction[:, -1, :-1] = bc_truth_enc[:, :xgrids-1]
    u_prediction[:, 1:, -1] = bc_truth_enc[:, xgrids-1:xgrids+ygrids-2]
    u_prediction[:, 0, 1:] = bc_truth_enc[:, xgrids+ygrids-2:2*xgrids+ygrids-3]
    u_prediction[:, :-1, 0] = bc_truth_enc[:, 2*xgrids+ygrids-3:]

    return u_prediction, ux_prediction


def evaluate(a_eval, bc_eval, eval_batch, net_u, net_ux,
             resolution, Ny, Nx, nsamples, predict_ux, device):
    '''
    Evaluate the model
    '''
    test_data = Data.TensorDataset(a_eval, bc_eval)
    test_loader = Data.DataLoader(test_data, eval_batch, shuffle=False)

    u_subdomain = torch.empty((0, resolution, resolution)).to(device)
    ux_subdomain = torch.empty((0, resolution-4, resolution-4)).to(device)
    net_u.eval()
    net_ux.eval()
    with torch.no_grad():
        for step, (input, boundary) in enumerate(test_loader):
            prediction_u = net_u(input, boundary, interp_mode='bilinear').squeeze(1)
            u_subdomain = torch.cat((u_subdomain, prediction_u), dim=0)

            if predict_ux:
                prediction = net_ux(input, boundary, interp_mode='bilinear').squeeze(1)[:, 2:-2, 2:-2]
                ux_subdomain = torch.cat((ux_subdomain, prediction), dim=0)

    u_subdomain = u_subdomain.reshape((Ny, Nx, nsamples, resolution, resolution))
    if predict_ux:
        ux_subdomain = ux_subdomain.reshape((Ny, Nx, nsamples, resolution-4, resolution-4))

    return u_subdomain, ux_subdomain