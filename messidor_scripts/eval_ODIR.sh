#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /itet-stor/garciaal/net_scratch/conda/etc/profile.d/conda.sh

# From here, it's just what you executed in qrsh
conda activate cutmix
# ... maybe other stuff here ...


cd ..

LR=1e-4
LAMBDA=0.25
BS=120
EPOCHS=100
DEPOCH = 50
FILE="mlc_ODIR_10p_LR-${LR}-lm_${LAMBDA}-bs-${BS}-ep-${EPOCHS}"

python baseline.py /datasets/ODIR/ ODIR exp/ODIR/$FILE -a resnet50 --gpu 0 -b $BS --base_lr $LR \
--epochs $EPOCHS  --decay_epoch 10 --adam --evaluate --num_class 8 --resume exp/ODIR/$FILE/checkpoint-ep70.pth.tar --custom_semi_predict

python average_result.py --filename $FILE
