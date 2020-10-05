#!/bin/bash
#$ -o qsub_output
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l gpu=1
#$ -l h_vmem=40G
source /itet-stor/garciaal/net_scratch/conda/etc/profile.d/conda.sh

# From here, it's just what you executed in qrsh
conda activate cutmix
# ... maybe other stuff here ...
 
# Now, you should use -u to get unbuffered output and "$@" for any arguments
sh train_ODIR.sh
