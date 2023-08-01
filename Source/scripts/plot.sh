#!/bin/bash

set -e

#CONFIG_FILE='./configs/ein_seld/seld.yaml'
# skip, relu, CBAMAttention, normal, Baluja # ours->skip (combine network)

python plots.py --dataset='CMU' --isCorrupted='corrupted' --module='normal' --apikey='21lzg3cz'

python plots.py --dataset='MTM' --isCorrupted='corrupted' --module='normal' --apikey='21lzg3cz'