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

from models.attention_MoE_mpaso import KVMemoryModel
from models.manager_mpaso import *


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

    parser.add_argument("--n-experts", type=int, default=2, help="number of experts")
    parser.add_argument("--gpu-id", type=int, default=0, help="id of GPU")
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
    network_str = f'mpaso_kv{num_feats}_MLP_dim{mlp_encoder_dim}_{num_hidden_layers}hLayers_keyDim{key_dim}_valDim{args.spatial_fdim}_{args.dim1d}line_{args.param_fdim}pDim_top{top_K}_M{n_experts}_alpha{args.alpha}_gateRes{gate_res}'
    
    if args.dropout != 0:
        network_str += '_dp'

    device = pytorch_device_config(args.gpu_id)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fh = open(os.path.join(args.root, "test", "names.txt"))
    filenames = []
    for line in fh:
        filenames.append(line.replace("\n", ""))

    params_arr = np.load(os.path.join(args.root, "test/params.npy"))
    coords = np.load(os.path.join(args.root, "sphereCoord.npy"))
    coords = coords.astype(np.float32)
    data_dicts = []
    for idx in range(len(filenames)):
        # params min [0.0, 300.0, 0.25, 100.0, 1]
        #        max [5.0, 1500.0, 1.0, 300.0, 384]
        params = np.array(params_arr[idx][1:])
        # print(params)
        params = (params.astype(np.float32) - np.array([0.0, 300.0, 0.25, 100.0], dtype=np.float32)) / \
                 np.array([5.0, 1200.0, .75, 200.0], dtype=np.float32)
        d = {'file_src': os.path.join(args.root, "test", filenames[idx]), 'params': params}
        data_dicts.append(d)

    lat_min, lat_max = -np.pi / 2, np.pi / 2
    coords[:,0] = (coords[:,0] - (lat_min + lat_max) / 2.0) / ((lat_max - lat_min) / 2.0)
    lon_min, lon_max = 0.0, np.pi * 2
    coords[:,1] = (coords[:,1] - (lon_min + lon_max) / 2.0) / ((lon_max - lon_min) / 2.0)
    depth_min, depth_max = 0.0, np.max(coords[:,2])
    coords[:,2] = (coords[:,2] - (depth_min + depth_max) / 2.0) / ((depth_max - depth_min) / 2.0)

    ensembleParam_dataset = EnsumbleParamDataset(data_dicts)
    ensembleParam_dataloader = DataLoader(ensembleParam_dataset, batch_size=1, shuffle=False, num_workers=0)

    #####################################################################################
    
    " Manager network "
    manager_net = Manager(resolution=gate_res, n_experts=n_experts)
    
    feat_shapes = np.ones(4, dtype=np.int32) * args.dim1d
    inr_fg = KVMemoryModel(feat_shapes, num_entries=num_feats, key_dim=key_dim, feature_dim_3d=args.spatial_fdim, 
                feature_dim_1d=args.param_fdim, top_K=top_K, chunk_size=chunk_size, 
                num_hidden_layers=num_hidden_layers, mlp_encoder_dim=mlp_encoder_dim,
                n_experts=n_experts, manager_net=manager_net)
    
    inr_fg.load_state_dict(torch.load(os.path.join("model_weights", network_str, "fg_model_" + network_str + '_'+ str(args.start_epoch) + ".pth"), 'cpu'))
    inr_fg.to(device)

    dmin = -1.93
    dmax = 30.36
    psnrs = []
    mds = []
    coords_torch = torch.from_numpy(coords)

    #####################################################################################

    with torch.no_grad():
        for param_idx, ensumbleParam_dict in enumerate(ensembleParam_dataloader):
            
            pred = np.zeros(len(coords), dtype=np.float32)
            
            params = ensumbleParam_dict['params'].reshape(1,4)
            params_batch = params.repeat(args.batch_size, 1)
            params_batch = params_batch.to(device)

            tstart = time.time()
            num_batches = math.ceil(len(coords) / args.batch_size)

            for field_idx in range(num_batches):
                coord_batch = coords_torch[field_idx*args.batch_size:(field_idx+1)*args.batch_size]
                if len(coord_batch) < args.batch_size:
                    params_batch = params.repeat(len(coord_batch), 1)
                    params_batch = params_batch.to(device)
                coord_batch = coord_batch.to(device)
                # ===================forward=====================
                model_output, probs = inr_fg(torch.cat((coord_batch, params_batch), 1))

                model_output = model_output.cpu().numpy().flatten().astype(np.float32)
                pred[field_idx*args.batch_size:(field_idx+1)*args.batch_size] = model_output
                
            tend = time.time()
            
            gt = ReadMPASOScalar(ensumbleParam_dict['file_src'][0])
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
            
            
            '''pred.tofile(args.dir_outputs + network_str + '_' + filenames[param_idx])'''
        print('<<<<<<<  PSNR = {0:.4f} >>>>>>>>>>'.format(np.mean(psnrs)))
        print('<<<<<<<  MD = {0:.4f} >>>>>>>>>>'.format(np.mean(mds)))


if __name__ == '__main__':
    main(parse_args())

