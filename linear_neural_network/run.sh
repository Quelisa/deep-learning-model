CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_file  ./xxx.txt \
    --dev_file    ./xxx.txt \
    --test_file   ./xxx.txt \
    --batch_size  32 \
    --epoch       3 \
    --lr          5e-5 \ 
    --input_dim   64 \
    --output_dim  16 \
    --predict     true