#!/bin/bash
#SBATCH -p dgx-a100-80g    
#SBATCH -G 1    
#SBATCH -t 3-0  
#SBATCH -J at 
#SBATCH -o ../log/stdout.%J   
#SBATCH -e ../log/stderr.%J 
#SBATCH --mail-type=ALL          # when you want to get notifications. You can select one from [BEGIN , END , FAIL , REQUEUE , ALL] 
#SBATCH --mail-user="matsukawa@mi.t.u-tokyo.ac.jp" # Email address which receives notifications







python main2.py > output.txt 2>&1
