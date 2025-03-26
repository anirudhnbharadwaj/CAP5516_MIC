#!/bin/bash

#SBATCH --job-name=BraTS_8

#SBATCH --partition=normal

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=64

#SBATCH --gpus=2

#SBATCH --mem=400G

#SBATCH --time=24:00:00

#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=an517162@ucf.edu

cd /lustre/fs1/home/cap5516.student5/CAP5516-MIC/A2/Task01

module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-11.8.0
module load gcc/gcc-12.2.0

source brain_seg_env/bin/activate

export OPENBLAS_NUM_THREADS=64

mkdir -p slurm_logs

JOB_ID=$SLURM_JOB_ID

# Run main.py 
echo "Starting main.py..." >> slurm_logs/slurm_${JOB_ID}.out
python src/main.py --config config/hyperparams.json 
echo "main.py completed." >> slurm_logs/slurm_${JOB_ID}.out

# Run optuna.py - Hyperparameter optimization with `n` trials
#echo "Starting run_optuna.py..." >> slurm_logs/slurm_${JOB_ID}.out
#python src/run_optuna.py --config config/hyperparams.json --n_trials 16
#echo "run_optuna.py completed." >> slurm_logs/slurm_${JOB_ID}.out

deactivate

echo "Job completed." >> slurm_logs/slurm_${JOB_ID}.out