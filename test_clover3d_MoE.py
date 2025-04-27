from utils import *
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import argparse
from tqdm import tqdm
import math
import pickle

from models.attention_MoE_cloverleaf import KVMemoryModel
from models.manager_cloverleaf import *


class EnsumbleParamDataset(Dataset):
    def __init__(self, params:list):
        ''' Input: param 1,2,3 and x,y,z '''
        self.params = params

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.params[idx]


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
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--weighted", action="store_true", default=False,
                        help="use weighted L1 Loss")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
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

    parser.add_argument("--n-experts", type=int, default=2, help="number of experts")
    parser.add_argument("--gpu-id", type=int, default=0, help="id of GPU")
    parser.add_argument("--gpu-ids", type=str, default="0,1", help="comma separated list of GPU ids to use")
    parser.add_argument("--mlp-encoder-dim", type=int, default=64, help="dimension of feat structure")
    parser.add_argument("--num-hidden-layers", type=int, default=2, help="number of hidden layers")
    
    parser.add_argument("--alpha", type=float, default=0.0, help="load balance loss weighting")
    parser.add_argument("--gate-res", type=int, default=16, help="resolution of gate")
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
    
    out_features = 1
    network_str = f'clover3d_kv{num_feats}_MLP_dim{mlp_encoder_dim}_{num_hidden_layers}hLayers_keyDim{key_dim}_valDim{args.spatial_fdim}_{args.dim1d}line_{args.param_fdim}pDim_top{top_K}_M{n_experts}_alpha{args.alpha}_gateRes{gate_res}'
    
    if args.dropout != 0:
        network_str += '_dp'

    # device = pytorch_device_config(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    device_ids = list(map(int, args.gpu_ids.split(',')))
    print(f"Using GPUs: {device_ids}")

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = os.listdir(args.root)
    files = sorted(files)
    arr_max = np.array([1.0, 2.0, 2.0, 3.5, 3.0, 7.0], dtype=np.float32)
    arr_min = np.array([0.01, 0.75, 0.51, 1.5, 1.5, 4.0], dtype=np.float32)
    training_dicts = []
    for fidx in range(len(files)):
        if files[fidx].endswith('.npy') and not files[fidx].endswith('params.npy'):
            sps = files[fidx].split('_')
            params_np = np.array([float(sps[1]), float(sps[2]), float(sps[3]), float(sps[4]), float(sps[5]), float(sps[6][:-4])], dtype=np.float32)
            params_np = (params_np - arr_min) / (arr_max - arr_min + 1e-8)
            d = {'file_src': os.path.join(args.root, files[fidx]), 'params': params_np}
            training_dicts.append(d)

    xcoords, ycoords, zcoords = np.linspace(-1,1,128), np.linspace(-1,1,128), np.linspace(-1,1,128)
    xv, yv, zv = np.meshgrid(xcoords, ycoords, zcoords, indexing='ij')
    xv, yv, zv = xv.flatten().reshape(-1,1), yv.flatten().reshape(-1,1), zv.flatten().reshape(-1,1)
    coords = np.hstack((xv, yv, zv)).astype(np.float32)

    ensembleParam_dataset = EnsumbleParamDataset(training_dicts)
    ensembleParam_dataloader = DataLoader(ensembleParam_dataset, batch_size=1, shuffle=False, num_workers=0)

    #####################################################################################
    
    " Manager network "
    manager_net = Manager(resolution=gate_res, n_experts=n_experts)
    
    feat_shapes = np.ones(6, dtype=np.int32) * args.dim1d
    inr_fg = KVMemoryModel(feat_shapes, num_entries=num_feats, key_dim=key_dim, feature_dim_3d=args.spatial_fdim, 
                feature_dim_1d=args.param_fdim, top_K=top_K, chunk_size=chunk_size, 
                num_hidden_layers=num_hidden_layers, mlp_encoder_dim=mlp_encoder_dim,
                n_experts=n_experts, manager_net=manager_net)
    
    inr_fg.load_state_dict(torch.load(os.path.join("model_weights", network_str, "fg_model_" + network_str + '_'+ str(args.start_epoch) + ".pth"), 'cpu'))
    # Use DataParallel for multi-GPU
    inr_fg = torch.nn.DataParallel(inr_fg, device_ids=device_ids)
    inr_fg.to(device)

    dmin = 0.36
    dmax = 41.92
    psnrs = []
    mds = []
    coords_torch = torch.from_numpy(coords)

    with torch.no_grad():
        for param_idx, ensumbleParam_dict in enumerate(ensembleParam_dataloader):
            pred = np.zeros(len(coords), dtype=np.float32)
            
            params = ensumbleParam_dict['params'].reshape(1,6)
            params_batch = params.repeat(args.batch_size, 1)
            params_batch = params_batch.to(device)

            tstart = time.time()
            num_batches = math.ceil(len(coords) / args.batch_size)

            for field_idx in range(num_batches):
                coord_batch = coords_torch[field_idx*args.batch_size:(field_idx+1)*args.batch_size]
                coord_batch = coord_batch.to(device)
                # ===================forward=====================
                model_output, probs = inr_fg(torch.cat((coord_batch, params_batch), 1))
                
                model_output = model_output.cpu().numpy().flatten().astype(np.float32)
                pred[field_idx*args.batch_size:(field_idx+1)*args.batch_size] = model_output
                
            tend = time.time()

            gt = ReadMPASOScalar(ensumbleParam_dict['file_src'][0]).reshape(-1,1)
            pred = pred * (dmax-dmin) + dmin
            mse = np.mean((gt - pred)**2)
            psnr = 20. * np.log10(dmax - dmin) - 10. * np.log10(mse)
            max_diff = abs(gt-pred)
            md = max_diff.max() / (dmax - dmin)
            psnrs.append(psnr)
            mds.append(md)
            print('Inference time: {0:.4f} , data: {1}'.format(tend-tstart, ensumbleParam_dict['file_src'][0]))
            print('PSNR = {0:.4f}, MSE = {1:.4f}'.format(psnr, mse))
            print('MD = {0:.4f}'.format(md))
            # pred.tofile(args.dir_outputs + network_str + '_' + files[param_idx])
        print('<<<<<<<  PSNR = {0:.4f} >>>>>>>>>>'.format(np.mean(psnrs)))
        print('<<<<<<<  MD = {0:.4f} >>>>>>>>>>'.format(np.mean(mds)))
    

if __name__ == '__main__':
    main(parse_args())