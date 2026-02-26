#!/bin/bash
#SBATCH --job-name=1-RDM-SVD-H
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mail-type=END,FAIL

module load foss/2020b
module load Eigen/3.3.9-GCCcore-10.2.0
module load R/4.2.0-foss-2020b

cd $SLURM_SUBMIT_DIR

Rscript run_model.R 1-RDM-SVD-H
