#!/bin/bash

echo "Activate environment"
source activate jalebi

echo "Start simulations"

echo "-------------------------------------------------------------"
echo "1 - input_length:5"
python train.py --input_length=5 --model_type='LSTM' --input_dim=10 --train_steps=5000

echo "-------------------------------------------------------------"
echo "1 - input_length:6"
python train.py --input_length=6 --model_type='LSTM' --input_dim=10 --train_steps=5000

echo "-------------------------------------------------------------"
echo "1 - input_length:7"
python train.py --input_length=7 --model_type='LSTM' --input_dim=10 --train_steps=5000

echo "-------------------------------------------------------------"
echo "1 - input_length:8"
python train.py --input_length=8 --model_type='LSTM' --input_dim=10 --train_steps=5000

echo "-------------------------------------------------------------"
echo "1 - input_length:9"
python train.py --input_length=9 --model_type='LSTM' --input_dim=10 --train_steps=5000

echo "-------------------------------------------------------------"
echo "1 - input_length:10"
python train.py --input_length=10 --model_type='LSTM' --input_dim=10 --train_steps=5000

echo "-------------------------------------------------------------"
echo "1 - input_length:11"
python train.py --input_length=11 --model_type='LSTM' --input_dim=10 --train_steps=5000

echo "-------------------------------------------------------------"
echo "1 - input_length:12"
python train.py --input_length=12 --model_type='LSTM' --input_dim=10 --train_steps=5000

echo "-------------------------------------------------------------"
echo "1 - input_length:13"
python train.py --input_length=13 --model_type='LSTM' --input_dim=10 --train_steps=5000

echo "-------------------------------------------------------------"
echo "1 - input_length:14"
python train.py --input_length=14 --model_type='LSTM' --input_dim=10 --train_steps=5000

echo "-------------------------------------------------------------"
echo "1 - input_length:15"
python train.py --input_length=15 --model_type='LSTM' --input_dim=10 --train_steps=5000
