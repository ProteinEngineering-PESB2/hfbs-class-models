#!/bin/bash
##---------------SLURM Parameters - NLHPC ----------------
#SBATCH -J group2
#SBATCH --mem=16gb
#SBATCH -o group2.out
#SBATCH -e group2.err
#SBATCH --cpus-per-task=8

#-----------------Módulos---------------------------
module load miniconda3
source activate p45_method

# ----------------Command--------------------------
python /home/dmedina/hydrophobin_class/source_code/classification_model/training_using_tpot.py /home/dmedina/hydrophobin_class/training_models/Group_2/encoding_data.csv /home/dmedina/hydrophobin_class/source_code/classification_model/group_2_optim.py
