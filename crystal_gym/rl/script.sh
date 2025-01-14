#!/bin/bash
#SBATCH --job-name=dft-BM-2190
#SBATCH --output=output/experiment-%A.%a.out
#SBATCH --error=error/experiment-%A.%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=12Gb
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --account=rrg-bengioy-ad

module load httpproxy/1.0
module load cuda/12.2
module load nvhpc/23.9
ml httpproxy
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NVHPC/Linux_x86_64/23.9/compilers/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NVHPC/Linux_x86_64/23.9/cuda/12.2/targets/x86_64-linux/lib"
source /home/pragov/envs/crystalgym/bin/activate
export OMP_NUM_THREADS=2

wandb login
# srun python ppo.py  wandb.wandb_project_name=$1 wandb.wandb_group=$2 exp.exp_name=$3 exp.seed=$4 env.seed=$5 env.index=$6 env.property=$7 env.p_hat=$8 qe.tstress=$9 qe.tprnfor=${10} qe.occupations=${11} algo.ent_coef=${12} algo.vf_coef=${13} algo.agent=CHGNetRL env.substitution=true  #env.mode=cubic-five
# srun python sac.py  wandb.wandb_project_name=$1 wandb.wandb_group=$2 exp.exp_name=$3 exp.seed=$4 env.seed=$5 env.index=$6 env.property=$7 env.p_hat=$8 qe.tstress=$9 qe.tprnfor=${10} qe.occupations=${11} algo.buffer_size=${12} algo.target_network_frequency=${13} env.vocab=large #algo.agent=CHGNetRL env.substitution=true 

# srun python dqn.py  wandb.wandb_project_name=$1 wandb.wandb_group=$2 exp.exp_name=$3 exp.seed=$4 env.seed=$5 env.index=$6 env.property=$7 env.p_hat=$8 qe.tstress=$9 qe.tprnfor=${10} qe.occupations=${11} algo.buffer_size=${12} algo.target_network_frequency=${13} qe.calculation=${14} algo.total_timesteps=${15} env.vocab=small #algo.replay_type=prioritized
srun python rainbow.py  wandb.wandb_project_name=$1 wandb.wandb_group=$2 exp.exp_name=$3 exp.seed=$4 env.seed=$5 env.index=$6 env.property=$7 env.p_hat=$8 qe.tstress=$9 qe.tprnfor=${10} qe.occupations=${11} algo.buffer_size=${12} algo.target_network_frequency=${13} algo.multi_step=${14} qe.calculation=${15} algo.total_timesteps=${16} env.vocab=small 
# srun python rainbow.py  wandb.wandb_project_name=$1 wandb.wandb_group=$2 exp.exp_name=$3 exp.seed=$4 env.seed=$5 env.index=$6 env.property=$7 env.p_hat=$8 qe.tstress=$9 qe.tprnfor=${10} qe.occupations=${11} algo.buffer_size=${12} algo.target_network_frequency=${13} algo.multi_step=${14} qe.calculation=${15} env.vocab=small #algo.replay_type=prioritized

