#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=human_eval
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/vast/mp5847/sa_%A_%a.out
#SBATCH --gres=gpu:a100:1

module purge

sleep $(( (RANDOM%10) + 1 ))
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo ${SLURM_ARRAY_TASK_ID}

singularity exec --nv \
  --overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate /scratch/mp5847/conda_environments/conda_pkgs/diffusion_ft; cd /home/mp5847/src/circumventing-concept-erasure/human_eval; export PYTHONPATH="$PYTHONPATH:$PWD";  \
    python3 generate.py '


