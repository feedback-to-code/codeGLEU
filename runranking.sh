#!/bin/sh
#SBATCH -A herbrich-student
#SBATCH --job-name=codegleu
#SBATCH --partition magic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=24:0:0

python runranking.py
