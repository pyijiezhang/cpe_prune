#!/bin/bash

Ts=(2.0 1.0 0.667 0.5 0.4 0.333 0.286)

for T in ${Ts[*]}; do 
    python experiments/train_lik.py --project_name="cpe" \
                                --wandb_mode="online" \
                                --seed=1 \
                                --dataset=fmnist \
                                --data_dir="fmnist" \
                                --dirty_lik="lenet" \
                                --likelihood="softmax" \
                                --augment=False \
                                --likelihood_temp=$T \
                                --temperature=1.0 \
                                --prior-scale=1.0 \
                                --sgld-epochs=100 \
                                --sgld-lr=1e-6 \
                                --momentum=0.99 \
                                --n-cycles=10 \
                                --n-samples=10 &  

    python experiments/train_lik.py --project_name="cpe" \
                                --wandb_mode="online" \
                                --seed=2 \
                                --dataset=fmnist \
                                --data_dir="fmnist" \
                                --dirty_lik="lenet" \
                                --likelihood="softmax" \
                                --augment=False \
                                --likelihood_temp=$T \
                                --temperature=1.0 \
                                --prior-scale=1.0 \
                                --sgld-epochs=100 \
                                --sgld-lr=1e-6 \
                                --momentum=0.99 \
                                --n-cycles=10 \
                                --n-samples=10 &

    python experiments/train_lik.py --project_name="cpe" \
                                --wandb_mode="online" \
                                --seed=3 \
                                --dataset=fmnist \
                                --data_dir="fmnist" \
                                --dirty_lik="lenet" \
                                --likelihood="softmax" \
                                --augment=False \
                                --likelihood_temp=$T \
                                --temperature=1.0 \
                                --prior-scale=1.0 \
                                --sgld-epochs=100 \
                                --sgld-lr=1e-6 \
                                --momentum=0.99 \
                                --n-cycles=10 \
                                --n-samples=10 &      
    wait
done