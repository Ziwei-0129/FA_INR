
############################################### MpasO ###############################################

python -u train_mpaso_MoE.py --root /research/nfs_chao_209/ziwei/mpas_graph/ --lr 0.0001 --sp-sr 1.0 --sf-sr 0.5 --load-batch 1 --batch-size 262144 --start-epoch 0 --epochs 50 --log-every 1 --check-every 5 --loss MSE --dim1d 10 --num-pairs 512 --spatial-fdim 16 --param-fdim 32 --dropout 0 --key-dim 16 --chunk-size 512 --mlp-encoder-dim 256 --num-hidden-layers 2 --gpu-id 0 --top-K 8 --lr-mlp 0.0001 --lr-gate 0.0001 --alpha 0.0 --n-experts 5 --gate-res 16



############################################### NYX ###############################################

python -u train_nyx_MoE.py --root /research/nfs_chao_209/ziwei/nyx/512/train/ --lr 0.0001 --sp-sr 1.0 --sf-sr 0.1 --load-batch 1 --batch-size 262144 --start-epoch 0 --epochs 50 --log-every 1 --check-every 5 --loss MSE --dim1d 16 --num-pairs 1024 --spatial-fdim 32 --param-fdim 64 --dropout 0 --key-dim 32 --chunk-size 1024 --mlp-encoder-dim 512 --lr-mlp 0.00001 --num-hidden-layers 4 --lr-gate 0.0001 --alpha 0.0 --n-experts 8 --top-K 16 --gate-res 16 --gpu-ids 0,1 --gpu-id 0



############################################### CloverLeaf ###############################################

python -u train_cloverleaf_MoE.py --root /research/nfs_chao_209/ziwei/cloverleaf3d/train/ --lr 0.0001 --sp-sr 1.0 --sf-sr 0.5 --load-batch 1 --batch-size 262144 --start-epoch 0 --epochs 55 --log-every 1 --check-every 5 --loss MSE --dim1d 10 --num-pairs 512 --spatial-fdim 16 --param-fdim 32 --dropout 0 --key-dim 16 --chunk-size 512 --mlp-encoder-dim 256 --num-hidden-layers 2 --gpu-id 0 --top-K 8 --lr-mlp 0.0001 --lr-gate 0.0001 --alpha 0.0 --n-experts 5 --gate-res 16


