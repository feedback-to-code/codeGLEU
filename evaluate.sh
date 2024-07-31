#!/bin/sh
#SBATCH -A herbrich-student
#SBATCH --job-name=codegleu
#SBATCH --partition magic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=8:0:0

python /hpi/fs00/home/fritz.darchinger/codeGLEU/evaluate.py
