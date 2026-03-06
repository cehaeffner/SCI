#!/bin/bash
#SBATCH --job-name=3-Joint-SVD-H
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=3-Joint-SVD-H-%j.out
#SBATCH --mail-type=END,FAIL

module load R/4.2.0-foss-2020b

cd ~/SCI/Stan
Rscript run_joint_model.R 3-Joint-SVD-H
