import torch
import torch.nn as nn

import argparse
import matplotlib.pyplot as plt
from timeit import default_timer
import os
import json
import yaml

from Model.tools import LpLoss
from Model.ppno import UNet

from DD.utils import load_exp_data, load_normalizer, load_model, init_a, init_u, partition_of_unity
from DD.DD import update_boundary_condition, reassemble_prediction, evaluate


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ############ Settings ############
    niter = args.niter
    predict_ux = args.predict_ux

    model_id_u = args.saved_name_u
    model_id_ux = args.saved_name_ux
    base_kernel = args.base_kernel
    trained_epoch = args.trained_epoch

    Nx = args.Nx
    Ny = args.Ny
    boundary_type = args.boundary_type
    resolution = args.resolution
    overlap_ratio = args.overlap_ratio
    nsamples = args.nsamples

    overlap_nodes = int((resolution-1)*overlap_ratio+1)
    nonoverlap_nodes = resolution - overlap_nodes

    xgrids = Nx*resolution-(Nx-1)*overlap_nodes
    ygrids = Ny*resolution-(Ny-1)*overlap_nodes

    eval_batch = 10

    id = 'r{}-{}by{}-{}-ux'.format(resolution, Nx, Ny, boundary_type)


    ############ Load Exp Data ############
    a_field, u_truth, ux_truth = load_exp_data(args.data_dir, resolution, Nx, Ny, boundary_type)


    ############ Parametize Normalizer ############
    a_normalizer, u_normalizer, ux_normalizer = load_normalizer(args.data_dir, resolution)


    ############ Define Model & Loss Func ############
    t1 = default_timer()
    net_u = UNet(sample_size=resolution, in_channels=2, out_channels=1, kernel_size=base_kernel)
    if predict_ux:
        net_ux = UNet(sample_size=resolution, in_channels=2, out_channels=1, kernel_size=base_kernel)

    loss_L2_func = LpLoss()
    loss_MSE_func = nn.MSELoss()


    ############ Load Model ############
    net_u = load_model(net_u, args.ckpt_dir, model_id_u, trained_epoch, device)
    net_u.eval()
    if predict_ux:
        net_ux = load_model(net_ux, args.ckpt_dir, model_id_ux, trained_epoch, device)
        net_ux.eval()


    ############ Initialize Problem ############
    '''
    a_subdomain: [Ny, Nx, nsamples, 1, resolution, resolution] ------ gpu
    a_eval: [Ny*Nx*nsamples, 1, resolution, resolution] ------------- gpu

    u_prediction: [nsamples, ygrids, xgrids] ------------------------ gpu
    u_prediction_save: [niter+1, nsamples, ygrids, xgrids] ---------- CPU
    u_truth: [nsamples, ygrids, xgrids] ----------------------------- CPU
    u_subdomain: [Ny, Nx, nsamples, resolution, resolution] --------- gpu

    ux_prediction: [nsamples, ygrids-4, xgrids-4] ------------------- gpu
    ux_prediction_save: [niter+1, nsamples, ygrids-4, xgrids-4] ----- CPU
    ux_truth: [nsamples, ygrids-4, xgrids-4] ------------------------ CPU
    ux_subdomain: [Ny, Nx, nsamples, 1, resolution-4, resolution-4] - gpu

    bc_subdomain: [Ny, Nx, nsamples, (resolution-1)*4] -------------- gpu
    bc_eval: [Ny*Nx*nsamples, (resolution-1)*4] --------------------- gpu

    ki_T_u, ki_R_u, ki_B_u, ki_L_u:  [nsamples, resolution, resolution] ------ gpu
    ki_T_ux,ki_R_ux,ki_B_ux,ki_L_ux: [nsamples, resolution-4, resolution-4] -- gpu
    '''
    ## a
    a_subdomain = init_a(a_field, Nx, Ny, nsamples, resolution, nonoverlap_nodes, a_normalizer, device)
    a_eval = a_subdomain.flatten(0,2)

    ## u
    bc_truth = torch.cat([u_truth[:, -1, :-1], u_truth[:, 1:, -1],
                          u_truth[:, 0, 1:], u_truth[:, :-1, 0]], dim=1)# [nsamples, (xgrids+ygrids-2)]
    bc_truth_enc = u_normalizer.encode(bc_truth).to(device)

    initial = init_u(bc_truth, xgrids, ygrids)

    # save the initial field
    u_prediction_save = torch.empty((niter+1, nsamples, ygrids, xgrids))
    u_prediction_save[0] = initial
    u_prediction = u_normalizer.encode(initial).to(device)

    L2_loss_list_u = [loss_L2_func(u_prediction_save[0], u_truth).item()]
    MSE_loss_list_u = [loss_MSE_func(u_prediction_save[0], u_truth).item()]

    if predict_ux:
        ux_prediction_save = torch.empty((niter+1, nsamples, ygrids-4, xgrids-4))
        ux_prediction_save[0] = torch.zeros((nsamples, ygrids-4, xgrids-4))
        ux_prediction = ux_normalizer.encode(ux_prediction_save[0]).to(device)

        L2_loss_list_ux = [loss_L2_func(ux_prediction_save[0], ux_truth).item()]
        MSE_loss_list_ux = [loss_MSE_func(ux_prediction_save[0], ux_truth).item()]

    ## partition of unity
    ki_T_u, ki_R_u, ki_B_u, ki_L_u, ki_T_ux, ki_R_ux, ki_B_ux, ki_L_ux = \
        partition_of_unity(nonoverlap_nodes, overlap_nodes, resolution, nsamples, device)

    t2 = default_timer()
    print('--Problem initialized, time: {:.4f}\n'.format(t2-t1))


    ############ Domain Decomposition ############
    print('----------------------------------')
    print('Nsample  : {}'.format(nsamples))
    print('Geometry : {} * {} rectangles (each subdomain with resolution {} * {}), meshgrid {} * {}'
          .format(Nx, Ny, resolution, resolution, xgrids, ygrids))
    print('Model    : {}, {}'.format(model_id_u, model_id_ux))
    print('Iteration: {}'.format(niter))
    print('----------------------------------')

    output_dir = './Results/'
    os.makedirs(output_dir, exist_ok=True)

    print('Start Domain Decomposition')
    u_normalizer.to(device)
    ux_normalizer.to(device)
    t1 = default_timer()
    for iter in range(niter):
        '''
            1) Update boundary condition: u_prediction -> bc_subdomain
            2) Evaluate the model: a_subdomain, bc_subdomain -> u_subdomain, and ux_subdomain
            3) Reassemble the prediction: u_subdomain -> u_prediction; ux_subdomain -> ux_prediction
            4) Repeat 1-3; Decode: u_prediction -> u_prediction_save; ux_prediction -> ux_prediction_save
        '''
        if (iter+1) % 10 == 0: print('    iter:',iter+1)
        t1_D = default_timer()

        ######### Update boundary condition #########
        bc_eval = update_boundary_condition(Nx, Ny, nsamples, nonoverlap_nodes, resolution, u_prediction, device)

        
        ######### Evaluate the model #########
        u_subdomain, ux_subdomain = evaluate(a_eval, bc_eval, eval_batch, net_u, net_ux,
                                            resolution, Ny, Nx, nsamples, predict_ux, device)


        ######### Reassamble Prediction #########
        u_prediction, ux_prediction = reassemble_prediction(
            nsamples, ygrids, xgrids, u_subdomain, ux_subdomain, resolution, nonoverlap_nodes,
            Ny, Nx, ki_T_u, ki_R_u, ki_B_u, ki_L_u, ki_T_ux, ki_R_ux, ki_B_ux, ki_L_ux,
            bc_truth_enc, predict_ux, device
        )
        

        # save and calculate the loss
        u_prediction_save[iter+1] = u_normalizer.decode(u_prediction).cpu()

        loss_L2_u = loss_L2_func(u_prediction_save[iter+1], u_truth).item()
        loss_MSE_u = loss_MSE_func(u_prediction_save[iter+1], u_truth).item()
        L2_loss_list_u.append(loss_L2_u)
        MSE_loss_list_u.append(loss_MSE_u)

        if predict_ux:
            ux_prediction_save[iter+1] = ux_normalizer.decode(ux_prediction).cpu()

            loss_L2_ux = loss_L2_func(ux_prediction_save[iter+1], ux_truth).item()
            loss_MSE_ux = loss_MSE_func(ux_prediction_save[iter+1], ux_truth).item()
            L2_loss_list_ux.append(loss_L2_ux)
            MSE_loss_list_ux.append(loss_MSE_ux)

        t2_D = default_timer()       
        if predict_ux:
            log_stats = {
                'iter': iter,
                'dt': '{:.4f}'.format(t2_D-t1_D), 
                'u(L2)': '{:.4e}'.format(loss_L2_u),
                'u(MSE)': '{:.4e}'.format(loss_MSE_u), 
                'ux(L2)': '{:.4e}'.format(loss_L2_ux),
                'ux(MSE)': '{:.4e}'.format(loss_MSE_ux)
            }
        else:
            log_stats = {
                'iter': iter,
                'dt': '{:.4f}'.format(t2_D-t1_D), 
                'u(L2)': '{:.4e}'.format(loss_L2_u),
                'u(MSE)': '{:.4e}'.format(loss_MSE_u)
            }
        with open(os.path.join(output_dir, "log-{}.txt".format(id)), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
        
    t2 = default_timer()
    print('----------------------------------')
    print('    Iter time: {:.4f}'.format(t2-t1))
    print('    L2 Loss :', loss_L2_u)
    print('    MSE Loss:', loss_MSE_u)
    print('    L2 Loss dx :', loss_L2_ux)
    print('    MSE Loss dx:', loss_MSE_ux)
    print('    ----------------------------------')

    # save the results
    torch.save({'prediction': u_prediction_save, 'target': u_truth},
               os.path.join(output_dir, 'u_prediction-{}.pt'.format(id)))
    if predict_ux:
        torch.save({'prediction': ux_prediction_save, 'target': ux_truth},
                os.path.join(output_dir, 'ux_prediction-{}.pt'.format(id)))


    # # print the smallest value of L2 loss and MSE loss and their corresponding iteration
    # if predict_ux:
    #     min_L2 = min(L2_loss_list_ux)
    #     min_MSE = min(MSE_loss_list_ux)
    #     min_L2_iter = L2_loss_list_ux.index(min_L2)
    #     min_MSE_iter = MSE_loss_list_ux.index(min_MSE)
    #     print('    dudx min L2 loss : {:.4e} at iteration {}'.format(min_L2, min_L2_iter))
    #     print('    dudx min MSE loss: {:.4e} at iteration {}'.format(min_MSE, min_MSE_iter))

    # min_L2_u = min(L2_loss_list_u)
    # min_MSE_u = min(MSE_loss_list_u)
    # min_L2_iter_u = L2_loss_list_u.index(min_L2_u)
    # min_MSE_iter_u = MSE_loss_list_u.index(min_MSE_u)
    # print('    u min L2 loss : {:.4e} at iteration {}'.format(min_L2_u, min_L2_iter_u))
    # print('    u min MSE loss: {:.4e} at iteration {}'.format(min_MSE_u, min_MSE_iter_u))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Name of the configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)
    
    main(args)