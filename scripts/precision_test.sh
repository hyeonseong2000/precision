#for ((int=1;int<=8;int++))
#do
#    for((frac=0;frac<=$[8-$int];frac++))
#    do
#        CUDA_VISIBLE_DEVICES=6 python precision.py --config configs/embedding.yaml --dtype "fxp" --exp $int --mant $frac --test --batch_size 200 
#    done
#done

CUDA_VISIBLE_DEVICES=6 python precision.py --config configs/embedding.yaml              \
                                           --dtype "fxp" --exp 8 --mant 23              \
                                           --test                                       \
                                           --batch_size 200                             \
                                           --mode "round"                               \
                                           --epoch 40                                   \
                                           --lr 0.0001                                  \
                                           #--resume_from "./saved_model/saved_model.pt" \



