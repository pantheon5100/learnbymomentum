python3 train.py \
        --gpu 0 \
        --dataset cifar100 \
        --data_dir ../../../../../data \
        --epoch 1000 \
        --bs 512 \
        --lr 5e-1 \
        --emb 128 \
        --method byol \
        --eval_every 20 \
        --optimizer sgd \
        --wandb \
        --name baseline-adam-lr0.5 \
        --project byol-momen 

