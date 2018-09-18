#!/bin/bash

echo "Activate environment"
source activate jalebi

echo "Start simulations"

echo "-------------------------------------------------------------"
echo "Default"
python train_mlp_pytorch.py

echo "-------------------------------------------------------------"
echo "dnn_hidden_units=500"
python train_mlp_pytorch.py --dnn_hidden_units=500 --max_steps=10000

echo "-------------------------------------------------------------"
echo "dnn_hidden_units=300,500,300"
python train_mlp_pytorch.py --dnn_hidden_units=300,500,300 --max_steps=10000

echo "-------------------------------------------------------------"
echo "dnn_hidden_units=500,500,500"
python train_mlp_pytorch.py --dnn_hidden_units=300,300,300 --max_steps=10000

echo "-------------------------------------------------------------"
echo "dnn_hidden_units=100,300,500,300,100"
python train_mlp_pytorch.py --dnn_hidden_units=100,300,500,300,100  --max_steps=10000

echo "-------------------------------------------------------------"
echo "learning_rate=0.001"
python train_mlp_pytorch.py --learning_rate=0.001 --max_steps=10000

echo "-------------------------------------------------------------"
echo "learning_rate=0.1"
python train_mlp_pytorch.py --learning_rate=0.1 --max_steps=10000

echo "-------------------------------------------------------------"
echo "batch_size=50"
python train_mlp_pytorch.py --batch_size=50 --max_steps=20000

echo "-------------------------------------------------------------"
echo "batch_size=500"
python train_mlp_pytorch.py --batch_size=500  --max_steps=10000
