#!/bin/bash

set -e

# python FinetuneBaluja.py --dataset='MTMdataset' --isCorrupted='uncorrupted'

# python FinetuneBaluja.py --dataset='MTMdataset' --isCorrupted='corrupted'



python Finetuning.py --dataset='CMU' 

python Finetuning.py --dataset='MTM' 

