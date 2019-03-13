#!/bin/bash
#SBATCH --job-name=PyXover
#SBATCH --account=j1010
#SBATCH --ntasks=8
#SBATCH --nodes=1
##SBATCH --time=00:05:00
##SBATCH --mem=2G

python3 PyXover.py 4

