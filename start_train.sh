python3 train.py \
        --gpu 4 \
        --dataset cifar100 \
        --data_dir /workspace/ssd2_4tb/data \
        --epoch 1000 \
        --bs 512 \
        --lr 5e-1 \
        --emb 128 \
        --method byol \
        --eval_every 20 \
        --optimizer adam \
        --wandb \
        --name baseline-adam-lr0.5 \
        --project byol-momen 

