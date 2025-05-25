#!/bin/bash
#SBATCH -p dgx-a100-80g    
#SBATCH -G 1     
#SBATCH -J check 

#SBATCH -o ./check/stdout.%J
#SBATCH -e ./check/stderr.%J
mkdir -p ./check
nvidia-smi
