python kfold_train.py \
    --experiment_no 6 \
    --embedding wikipedia \
    --output_dim 3 \
    --embedding_dim 300 \
    --hidden_dim 256 \
    --num_rnn_layers 1 \
    --num_linear_layers 1 \
    --rnn_dropout 0.3 \
    --linear_dropout 0.5 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --random_seed 7 \
    --kfold 5 \
    --epoch 50 \
    --batch_size 16