#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /itet-stor/garciaal/net_scratch/conda/etc/profile.d/conda.sh shell.bash hook

# From here, it's just what you executed in qrsh
conda activate cutmix
# ... maybe other stuff here ...
 

cd ..
	
LR=$1
LAMBDA=$2
BS=$3
EPOCHS=$4
DEPOCH=$5
FILE="/mlc_ODIR_10p_LR-${LR}-lm_${LAMBDA}-bs-${BS}-ep-${EPOCHS}"


    
python baseline.py ./data/ ODIR exp/ODIR/$FILE -a resnet50 \
--gpu 0 -b $BS --base_lr $LR --pretrained --epochs $EPOCHS  --decay_epoch $DEPOCH --num_class 8 --adam \
--lambda_value $LAMBDA
