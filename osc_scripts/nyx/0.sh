#!/bin/bash
#
#SBATCH --job-name=Surrogate_MOE
#SBATCH --output="osc_logs/nyx/%j.txt"
#SBATCH --signal=USR1@20
#SBATCH --nodes=1 --gpus-per-node 1 --mem-per-gpu 128GB --cpus-per-gpu 10
#SBATCH --time=23:59:00
#SBATCH --account=PAS0027

# prep software and move to directory
source ~/ascend.bashrc
cd $HOME/project/alsrn ## change to your project directory

# load python environment
conda deactivate
conda activate vis11

# execute your job
FREE_PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d: -f2 | sort -u) | shuf | head -n 1)
echo "Found free port: $FREE_PORT"

python start_jobs.py --free_port $FREE_PORT --script train_srn.py \
--accelerate_config /users/PAS0027/xiong336/project/alsrn/configs/hdinr/accl/one.yaml \
--config /users/PAS0027/xiong336/project/alsrn/configs/hdinr/single_gpu/nyx/nyx_baselines_100_s2x.yaml --config_id 0

python start_jobs.py --free_port $FREE_PORT --script test_srn.py \
--accelerate_config /users/PAS0027/xiong336/project/alsrn/configs/hdinr/accl/one.yaml \
--config /users/PAS0027/xiong336/project/alsrn/configs/hdinr/single_gpu/nyx/nyx_baselines_100_s2x.yaml --config_id 0
