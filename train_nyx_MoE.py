from utils import *
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm
import math
import yaml
from models.modules import fwd_mlp, fwd_mmgn_cond, fwd_mmgn_idx
from models.baselines.kplane import KPlaneField
from models.baselines.coordnet import CoordNet
from models.baselines.mmgn import MMGNet
from models.attention_MoE_cloverleaf import KVMemoryModel
from models.manager_cloverleaf import *


# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--dir-weights", required=False, type=str,
                        help="model weights path")
    parser.add_argument("--dir-outputs", required=False, type=str,
                        help="directory for any outputs (ex: images)")
    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")
    parser.add_argument("--dsp", type=int, default=3,
                        help="dimensions of the simulation parameters (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--sp-sr", type=float, default=0.3,
                        help="simulation parameter sampling rate (default: 0.2)")
    parser.add_argument("--sf-sr", type=float, default=0.05,
                        help="scalar field sampling rate (default: 0.02)")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="beta1 of Adam (default: 0.0)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--load-batch", type=int, default=1,
                        help="batch size for loading (default: 1)")
    parser.add_argument("--batch-size", type=int, default=65536,
                        help="batch size for training (default: 1)")
    parser.add_argument("--weighted", action="store_true", default=False,
                        help="use weighted L1 Loss")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of epochs to train (default: 10000)")
    parser.add_argument("--log-every", type=int, default=1,
                        help="log training status every given number of batches (default: 1)")
    parser.add_argument("--check-every", type=int, default=2,
                        help="save checkpoint every given number of epochs (default: 2)")
    parser.add_argument("--loss", type=str, default='MSE',
                        help="loss function for training (default: MSE)")
    parser.add_argument("--dim1d", type=int, default=32,
                        help="dimension of 1D line for parameter domain")
    parser.add_argument("--num-pairs", type=int, default=2048,      # number of KV spatial feature pairs
                        help="number of KV spatial feature pairs")
    parser.add_argument("--key-dim", type=int, default=3,           # key dim
                        help="dimension of feature for spatial domain in feature grids")
    parser.add_argument("--top-K", type=int, default=16,            # number of top_K keys
                        help="number of top_K keys")
    parser.add_argument("--chunk-size", type=int, default=256,      # size of chunked kv paris
                        help="size of chunked kv paris")
    parser.add_argument("--spatial-fdim", type=int, default=8,      # value dim
                        help="dimension of feature for spatial domain in feature grids")
    parser.add_argument("--param-fdim", type=int, default=8,
                        help="dimension of feature for parameter domain in feature grids")
    parser.add_argument("--dropout", type=int, default=0,
                        help="using dropout layer in MLP, 0: No, other: Yes (default: 0)")
    # Mixture-of-Experts
    parser.add_argument("--n-experts", type=int, default=2, help="number of experts")
    parser.add_argument("--gpu-id", type=int, default=0, help="id of GPU")
    parser.add_argument("--gpu-ids", type=str, default="0", help="comma separated list of GPU ids to use")
    parser.add_argument("--mlp-encoder-dim", type=int, default=64, help="dimension of feat structure")
    parser.add_argument("--num-hidden-layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--lr-mlp", type=float, default=1e-4, help="encoder MLP learning rate (default: 1e-4)")
    parser.add_argument("--lr-gate", type=float, default=1e-4, help="gate MLP learning rate (default: 1e-4)")
    
    parser.add_argument("--alpha", type=float, default=0.0, help="load balance loss weighting")
    parser.add_argument("--gate-res", type=int, default=16, help="resolution of gate")
    
    
    ################## baseline arguments ##################
    parser.add_argument("--base_model", type=str, choices=["coordnet", "kplane", "mmgn"],
                        default=None, help="baseline model to use")
    
    parser.add_argument("--base_model_config", type=str, required=False,
                        default='configs/nyx/default_models.yaml',
                        help="path to config file (optional)")
    
    return parser.parse_args()


def main(args):
    # log hyperparameters
    print(args)
    num_feats = args.num_pairs
    key_dim = args.key_dim
    top_K = args.top_K
    chunk_size = args.chunk_size
    
    n_experts = args.n_experts
    mlp_encoder_dim = args.mlp_encoder_dim
    num_hidden_layers = args.num_hidden_layers
    gate_res = args.gate_res

    nEnsemble = 4
    data_size = 512**3
    num_sf_batches = math.ceil(nEnsemble * data_size * args.sf_sr / args.batch_size)

    network_str = f'nyx_kv{num_feats}_MLP_dim{mlp_encoder_dim}_{num_hidden_layers}hLayers_keyDim{key_dim}_valDim{args.spatial_fdim}_{args.dim1d}line_{args.param_fdim}pDim_top{top_K}_M{n_experts}_alpha{args.alpha}_gateRes{gate_res}_k2'
    lr_mlp = args.lr_mlp
    lr_gate = args.lr_gate
   
    args.dir_weights = os.path.join("model_weights", network_str)
    args.dir_outputs = os.path.join("outputs", network_str)
    if not os.path.exists(args.dir_weights):
        os.mkdir(args.dir_weights)
        os.mkdir(args.dir_outputs)
    
    
    if args.dropout != 0:
        network_str += '_dp'

    # Device setup
    # device = pytorch_device_config(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    device_ids = list(map(int, args.gpu_ids.split(',')))
    print(f"Using GPUs: {device_ids}")


    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = os.listdir(args.root)
    files = sorted(files)
    training_dicts = []
    for fidx in range(len(files)):
        if files[fidx].endswith('bin'):
            sps = files[fidx].split('_')
            # params min [0.12, 0.0215, 0.55]
            #        max [0.155, 0.0235, 0.85]
            params_np = np.array([float(sps[1]), float(sps[2]), float(sps[3][:-4])], dtype=np.float32)
            params_np = (params_np - np.array([0.12, 0.0215, 0.55], dtype=np.float32)) / np.array([0.035, 0.002, 0.3], dtype=np.float32)
            d = {'file_src': os.path.join(args.root, files[fidx]), 'params': params_np}
            training_dicts.append(d)

    xcoords, ycoords, zcoords = np.linspace(-1,1,512), np.linspace(-1,1,512), np.linspace(-1,1,512)
    xv, yv, zv = np.meshgrid(xcoords, ycoords, zcoords, indexing='ij')
    xv, yv, zv = xv.flatten().reshape(-1,1), yv.flatten().reshape(-1,1), zv.flatten().reshape(-1,1)
    coords = np.hstack((xv, yv, zv)).astype(np.float32)

    #####################################################################################
    
    fwd_fn = None
    if args.base_model is not None:
        print(f"Using base model: {args.base_model}")
        with open(args.base_model_config, 'r') as f:
            base_model_config = yaml.full_load(f)
        if args.base_model == "coordnet":
            inr_fg = CoordNet(**base_model_config[args.base_model])
            fwd_fn = fwd_mlp
        elif args.base_model == "kplane":
            inr_fg = KPlaneField(**base_model_config[args.base_model])
            fwd_fn = fwd_mlp
        elif args.base_model == "mmgn":
            inr_fg = MMGNet(**base_model_config[args.base_model])
            cond_bank = torch.tensor([d['params'] for d in training_dicts], dtype=torch.float32).to(device)
            inr_fg.set_cond_bank(cond_bank)
            fwd_fn = fwd_mmgn_idx
        else:
            raise ValueError(f"Unknown base model: {args.base_model}")
        
    else:
        " Manager network "
        manager_net = Manager(resolution=gate_res, n_experts=n_experts)
        
        feat_shapes = np.ones(3, dtype=np.int32) * args.dim1d
        inr_fg = KVMemoryModel(feat_shapes, num_entries=num_feats, key_dim=key_dim, feature_dim_3d=args.spatial_fdim, 
                    feature_dim_1d=args.param_fdim, top_K=top_K, chunk_size=chunk_size, 
                    num_hidden_layers=num_hidden_layers, mlp_encoder_dim=mlp_encoder_dim,
                    n_experts=n_experts, manager_net=manager_net)
 

    # Use DataParallel for multi-GPU
    inr_fg = torch.nn.DataParallel(inr_fg, device_ids=device_ids)
    inr_fg.to(device)
    
    if args.start_epoch > 0:
        inr_fg.module.load_state_dict(torch.load(os.path.join(args.dir_weights, network_str, "fg_model_" + network_str + '_'+ str(args.start_epoch) + ".pth"), 'cpu'))

    # Set optimizer:
    if args.base_model is not None:
        optimizer = torch.optim.Adam(inr_fg.module.parameters(), lr=args.lr)
    else:
        encoder_mlp_params = set(inr_fg.module.encoder_mlp_list.parameters())
        gating_mlp_params = set(inr_fg.module.manager_net.parameters())
        other_parameters = (param for param in inr_fg.parameters() if param not in encoder_mlp_params and \
                            param not in gating_mlp_params)
        optimizer = torch.optim.Adam([
            {'params': inr_fg.module.manager_net.parameters(), 'lr': lr_gate},
            {'params': inr_fg.module.encoder_mlp_list.parameters(), 'lr': lr_mlp},
            {'params': other_parameters, 'lr': args.lr},
        ])
    
    if args.loss == 'L1':
        print('Use L1 Loss')
        criterion = torch.nn.L1Loss()
    else:
        print('Use MSE Loss')
        criterion = torch.nn.MSELoss()

    #####################################################################################

    losses = []

    dmin = 8.6
    dmax = 13.6
    num_bins = 10
    cluster_size = data_size // np.power(2, 24) # for multinomial, number of categories cannot exceed 2^24
    bin_width = 1.0 / num_bins
    max_binidx_f = float(num_bins-1)
    batch_size_per_field = args.batch_size // nEnsemble
    nEnsembleGroups_per_epoch = len(training_dicts) // nEnsemble

    coords_torch = torch.from_numpy(coords)

    sfimps_np = np.load(os.path.join('outputs', 'nyx_ensemble_member_importances.npy'))
    sfimps = torch.from_numpy(sfimps_np)

    #####################################################################################

    def imp_func(data, minval, maxval, bw, maxidx):
        freq = None
        nBlocks = 16
        block_size = data_size // nBlocks
        for bidx in range(nBlocks):
            block_freq = torch.histc(data[bidx*block_size:(bidx+1)*block_size], bins=num_bins, min=minval, max=maxval).type(torch.long)
            if freq is None:
                freq = block_freq
            else:
                freq += block_freq
        freq = freq.type(torch.double)
        importance = 1. / freq
        importance_idx = torch.clamp((data - minval) / bw, min=0.0, max=maxidx).type(torch.long)
        return importance, importance_idx

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        print('epoch {0}'.format(epoch+1))
        
        total_loss = 0
        e_rndidx = torch.multinomial(sfimps, nEnsembleGroups_per_epoch * nEnsemble, replacement=True)
        for egidx in range(nEnsembleGroups_per_epoch):
            tstart = time.time()
            scalar_fields = []
            sample_weights_arr = []
            params_batch = None
            errsum = 0
            # Load and compute importance map
            batch_e_indices = []
            for eidx in range(nEnsemble):
                batch_e_indices.append(e_rndidx[egidx*nEnsemble + eidx])
                curr_scalar_field = ReadScalarBinary(training_dicts[e_rndidx[egidx*nEnsemble + eidx]]['file_src'])
                curr_scalar_field = (curr_scalar_field-dmin) / (dmax-dmin)
                curr_scalar_field = torch.from_numpy(curr_scalar_field)
                curr_params = training_dicts[e_rndidx[egidx*nEnsemble + eidx]]['params'].reshape(1,3)
                curr_params = torch.from_numpy(curr_params)
                curr_params_batch = curr_params.repeat(batch_size_per_field, 1)
                if params_batch is None:
                    params_batch = curr_params_batch
                else:
                    params_batch = torch.cat((params_batch, curr_params_batch), 0)
                curr_imp, curr_impidx = imp_func(curr_scalar_field, 0.0, 1.0, bin_width, max_binidx_f)
                curr_sample_weights = curr_imp[curr_impidx].reshape(-1, cluster_size).sum(1)
                
                scalar_fields.append(curr_scalar_field)
                sample_weights_arr.append(curr_sample_weights)
            batch_e_indices = torch.stack(batch_e_indices, dim=0)
            params_batch = params_batch.to(device)
            # Train
            for field_idx in range(num_sf_batches):
                coord_batch = None
                value_batch = None
                for eidx in range(nEnsemble):
                    #####
                    rnd_idx = torch.multinomial(sample_weights_arr[eidx], batch_size_per_field, replacement=True)
                    rnd_idx = rnd_idx * cluster_size + torch.randint(high=cluster_size, size=rnd_idx.shape)
                    ######
                    if coord_batch is None:
                        coord_batch, value_batch = coords_torch[rnd_idx], scalar_fields[eidx][rnd_idx]
                    else:
                        coord_batch, value_batch = torch.cat((coord_batch, coords_torch[rnd_idx]), 0), torch.cat((value_batch, scalar_fields[eidx][rnd_idx]), 0)
                value_batch = value_batch.reshape(len(value_batch), 1)
                coord_batch = coord_batch.to(device)
                value_batch = value_batch.to(device)
                # ===================forward=====================
                if args.base_model is None: # if using MoE
                    model_output, probs = inr_fg(torch.cat((coord_batch, params_batch), 1), tau=1.0)
                else:
                    model_output = fwd_fn(
                        inr_fg,
                        coord_batch.view(nEnsemble, -1, 3),
                        params_batch.view(nEnsemble, -1, 3),
                        batch_e_indices
                    )
                loss = criterion(model_output, value_batch)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_mean_loss = loss.data.cpu().numpy()
                errsum += batch_mean_loss * nEnsemble * batch_size_per_field
                total_loss += batch_mean_loss
                
            tend = time.time()
            mse = errsum / (nEnsemble * batch_size_per_field * num_sf_batches)
            curr_psnr = - 10. * np.log10(mse)
            print('Training time: {0:.4f} for {1} data points x {2} batches, approx PSNR = {3:.4f}'\
                  .format(tend-tstart, nEnsemble * batch_size_per_field, num_sf_batches, curr_psnr))
        losses.append(total_loss)
        if (epoch+1) % args.log_every == 0:
            print('epoch {0}, loss = {1}'.format(epoch+1, total_loss))
            print("====> Epoch: {0} Average {1} loss: {2:.4f}".format(epoch+1, args.loss, total_loss / (nEnsembleGroups_per_epoch * nEnsemble)))
            plt.plot(losses)

            plt.savefig(os.path.join(args.dir_outputs, 'nyx_fg_inr_loss_' + network_str + '.jpg'))
            plt.clf()

        if (epoch+1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            torch.save(inr_fg.module.state_dict(), 
                        os.path.join(args.dir_weights, "fg_model_" + network_str + '_'+ str(epoch+1) + ".pth"))
            

if __name__ == '__main__':
    main(parse_args())
    