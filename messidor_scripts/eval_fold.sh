cd ..
max=1
for i in `seq 1 $max`
do
    NUM="${var}$i"
    FILE="${var}/multi_CAN_lamda25_noshare_wpre_simpleaug_10fold$NUM""_1000"
    FOLD="${var}fold$NUM"
    python baseline.py /raid/li/datasets/MESSIDOR/ missidor exp/MESSIDOR/$FILE --fold_name $FOLD -a resnet50 --gpu 8 -b 8 --base_lr 3e-4 --pretrained --epochs 20  --decay_epoch 10  --multitask --crossCBAM --choice both --adam --evaluate --resume exp/MESSIDOR/$FILE/model_converge.pth.tar
done
python average_result.py --filename multi_CAN_lamda25_noshare_wpre_simpleaug_10fold
