#!/bin/bash

echo "Run Train" 
python ./train.py --steps_param 0 --epochs 20  --batch_size 128 --minibatch 64 --lr 0.0010646  --device cuda:0



