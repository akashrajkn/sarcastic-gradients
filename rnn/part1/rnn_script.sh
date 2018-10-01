#!/bin/bash

echo "Activate environment"
source activate jalebi

echo "Start simulations"

echo "-------------------------------------------------------------"
echo "1 - input_length:5"
python train.py --input_length=10 --model_type='RNN' --input_dim=10 --train_steps=3000 --learning_rate=0.001

echo "-------------------------------------------------------------"
echo "1 - input_length:6"
python train.py --input_length=10 --model_type='RNN' --input_dim=10 --train_steps=3000 --learning_rate=0.01

echo "-------------------------------------------------------------"
echo "1 - input_length:7"
python train.py --input_length=10 --model_type='RNN' --input_dim=10 --train_steps=3000 --learning_rate=0.025

echo "-------------------------------------------------------------"
echo "1 - input_length:8"
python train.py --input_length=10 --model_type='RNN' --input_dim=10 --train_steps=3000 --learning_rate=0.0001
