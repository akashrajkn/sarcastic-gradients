#!/bin/bash

echo "Activate environment"
source activate jalebi

echo "Start simulations"

echo "-------------------------------------------------------------"
echo "1 - input_length:5"
python train.py --input_length=5

echo "-------------------------------------------------------------"
echo "1 - input_length:6"
python train.py --input_length=6

echo "-------------------------------------------------------------"
echo "1 - input_length:7"
python train.py --input_length=7

echo "-------------------------------------------------------------"
echo "1 - input_length:8"
python train.py --input_length=8

echo "-------------------------------------------------------------"
echo "1 - input_length:9"
python train.py --input_length=9

echo "-------------------------------------------------------------"
echo "1 - input_length:10"
python train.py --input_length=10

echo "-------------------------------------------------------------"
echo "1 - input_length:11"
python train.py --input_length=11

echo "-------------------------------------------------------------"
echo "1 - input_length:12"
python train.py --input_length=12

echo "-------------------------------------------------------------"
echo "1 - input_length:13"
python train.py --input_length=13

echo "-------------------------------------------------------------"
echo "1 - input_length:14"
python train.py --input_length=14

echo "-------------------------------------------------------------"
echo "1 - input_length:15"
python train.py --input_length=15
