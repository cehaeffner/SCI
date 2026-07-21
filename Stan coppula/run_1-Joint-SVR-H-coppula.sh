#!/bin/bash
#SBATCH --job-name=1-Cop-SVR-H
#SBATCH --cpus-per-task=4
#SBATCH --partition=psych_day
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=1-Joint-SVR-H-coppula-%j.out
#SBATCH --mail-type=END,FAIL

module load R/4.2.0-foss-2020b

cd "$HOME/SCI/Stan coppula"
Rscript run_model-coppula.R 1-Joint-SVR-H
