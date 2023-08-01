#!/bin/bash

set -e

# python FinetuneBaluja.py --dataset='MTMdataset' --isCorrupted='uncorrupted'

# python FinetuneBaluja.py --dataset='MTMdataset' --isCorrupted='corrupted'

python Train_CMU.py --dataset='CMUdataset' --isCorrupted='uncorrupted' --module='Baluja'

python Train_CMU.py --dataset='CMUdataset' --isCorrupted='corrupted' --module='Baluja'