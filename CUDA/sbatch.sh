#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments.  Uncomment
#       "##SBATCH" line means to remove one # and start with #SBATCH to be a
#       SLURM command or statement.

#SBATCH -J try #Slurm job name
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err

# Set the maximum runtime, uncomment if you need it
##SBATCH -t 48:00:00 #Maximum runtime of 48 hours

# Enable email notificaitons when job begins and ends, uncomment if you need it
##SBATCH –-mail-user=hzhangaq@ust.hk #Update your email address
##SBATCH –-mail-type=begin
##SBATCH –-mail-type=end

# Choose partition (queue)
#SBATCH -p gpu-k

# To use 2 cpu cores in a node, uncomment the statement below
##SBATCH -N 1 -n 2

# To use 1 cpu core and 1 gpu device, uncomment the statement below
#SBATCH -N 1 -n 1 --gres=gpu:1

./executable
