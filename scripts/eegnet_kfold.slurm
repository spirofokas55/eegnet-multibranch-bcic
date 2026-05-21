#!/bin/bash
#SBATCH --job-name=eegnet_kfold
#SBATCH --partition=work1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sfokas/eegnet_jobs/slurm_kfold_%A_%a.out
#SBATCH --array=0-7

# Subjects: 8 (A04T excluded — EOG recording inconsistency)
SUBJECTS=(A01T A02T A03T A05T A06T A07T A08T A09T)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=/scratch/$USER/BCICIV_2a_gdf
OUT_DIR=/scratch/$USER/eegnet_kfold_runs
SCRIPT_DIR=$HOME/eegnet-multibranch-bcic/scripts

module load anaconda3
source activate yolo

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[$SLURM_ARRAY_TASK_ID] Subject=$SUBJ  Node=$(hostname)  GPU=$CUDA_VISIBLE_DEVICES"

python "$SCRIPT_DIR/train_eegnet_kfold.py" \
  --data_dir  "$DATA_DIR" \
  --out_dir   "$OUT_DIR"  \
  --subject   "$SUBJ"     \
  --seed      42          \
  --num_workers 4         \
  --pin_memory            \
  --use_amp
