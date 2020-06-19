cd ..
max=1
for i in `seq 1 $max`
do
    NUM="${var}$i"
    FILE="${var}/multi_CAN_lamda25_noshare_wpre_simpleaug_10fold$NUM""_1000"
    FOLD="${var}fold$NUM"
    python baseline.py ./data/ ODIR exp/ODIR/$FILE --fold_name $FOLD -a resnet50 \
    --gpu 0 -b 8 --base_lr 3e-4 --pretrained --epochs 2  --decay_epoch 1 --num_class 2 --adam \
    --crossCBAM --choice both --lambda_value 0.25
done
