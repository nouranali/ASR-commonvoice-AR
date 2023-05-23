#!/bin/bash

#SBATCH --job-name=asr_test
#SBATCH --output=/home/nouran.ali/output/%j%x.out 
#SBATCH --error=/home/nouran.ali/output/%j%x.err 
#SBATCH --time=24:00:00  
#SBATCH --nodes=1      
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:4

python /home/nouran.ali/commonvoice/code/test.py