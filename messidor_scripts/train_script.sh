#!/bin/bash
    
LR=1e-4
EPOCHS=100
LAMBDA=0.25
BS=20
DEPOCH=50

JOB_NAME="ODIR_lr=${LR},lm=${LAMBDA},bs=${BS},ep=${EPOCHS}"

qsub -N $JOB_NAME train_ODIR.sh $LR $LAMBDA $BS $EPOCHS $DEPOCH

LR=1e-4
EPOCHS=100
LAMBDA=0.25
BS=40
DEPOCH=50

JOB_NAME="ODIR_lr=${LR},lm=${LAMBDA},bs=${BS},ep=${EPOCHS}"

qsub -N $JOB_NAME train_ODIR.sh $LR $LAMBDA $BS $EPOCHS $DEPOCH

LR=1e-4
EPOCHS=100
LAMBDA=0.25
BS=125
DEPOCH=50

JOB_NAME="ODIR_lr=${LR},lm=${LAMBDA},bs=${BS},ep=${EPOCHS}"

qsub -N $JOB_NAME train_ODIR.sh $LR $LAMBDA $BS $EPOCHS $DEPOCH

LR=1e-3
EPOCHS=100
LAMBDA=0.25
BS=250
DEPOCH=50

JOB_NAME="ODIR_lr=${LR},lm=${LAMBDA},bs=${BS},ep=${EPOCHS}"

qsub -N $JOB_NAME train_ODIR.sh $LR $LAMBDA $BS $EPOCHS $DEPOCH

LR=1e-3
EPOCHS=100
LAMBDA=0.25
BS=250
DEPOCH=50

JOB_NAME="ODIR_lr=${LR},lm=${LAMBDA},bs=${BS},ep=${EPOCHS}"

qsub -N $JOB_NAME train_ODIR.sh $LR $LAMBDA $BS $EPOCHS $DEPOCH

LR=1e-3
EPOCHS=100
LAMBDA=0.25
BS=250
DEPOCH=50

JOB_NAME="ODIR_lr=${LR},lm=${LAMBDA},bs=${BS},ep=${EPOCHS}"

qsub -N $JOB_NAME train_ODIR.sh $LR $LAMBDA $BS $EPOCHS $DEPOCH

LR=1e-3
EPOCHS=100
LAMBDA=0.25
BS=300
DEPOCH=50

JOB_NAME="ODIR_lr=${LR},lm=${LAMBDA},bs=${BS},ep=${EPOCHS}"

qsub -N $JOB_NAME train_ODIR.sh $LR $LAMBDA $BS $EPOCHS $DEPOCH

LR=5e-4
EPOCHS=100
LAMBDA=0.25
BS=250
DEPOCH=50

JOB_NAME="ODIR_lr=${LR},lm=${LAMBDA},bs=${BS},ep=${EPOCHS}"

qsub -N $JOB_NAME train_ODIR.sh $LR $LAMBDA $BS $EPOCHS $DEPOCH

LR=5e-4
EPOCHS=150
LAMBDA=0.25
BS=250
DEPOCH=75

JOB_NAME="ODIR_lr=${LR},lm=${LAMBDA},bs=${BS},ep=${EPOCHS}"

qsub -N $JOB_NAME train_ODIR.sh $LR $LAMBDA $BS $EPOCHS $DEPOCH
