#!/bin/bash

set -e

#CONFIG_FILE='./configs/ein_seld/seld.yaml'
# skip, relu, CBAMAttention, normal, Baluja # ours->skip (combine network)

python plot_corrupted_PSNR.py --dataset='MTM' --isCorrupted='corrupted' --module='normal'

python plot_corrupted_PSNR.py --dataset='CMU' --isCorrupted='corrupted' --module='normal'
