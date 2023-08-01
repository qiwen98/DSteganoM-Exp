#!/bin/bash

set -e

#CONFIG_FILE='./configs/ein_seld/seld.yaml'
# normal, relu, skip, Baluja, CBAMAttention, OursGated

#python test.py --dataset='MTM' --isCorrupted='uncorrupted' --module='normal' --apikey='21lzg3cz'

#python test.py --dataset='MTM' --isCorrupted='uncorrupted' --module='skip' --apikey='25r07hc8'
#
#python test.py --dataset='MTM' --isCorrupted='uncorrupted' --module='relu' --apikey='1ncyp1r4'
#
#python test.py --dataset='MTM' --isCorrupted='uncorrupted' --module='CBAMAttention' --apikey='3aeamup4'
#
#python test.py --dataset='MTM' --isCorrupted='uncorrupted' --module='Baluja' --apikey='2cfbodnn'

#python test.py --dataset='MTM' --isCorrupted='uncorrupted' --module='OursGated' --apikey='1oywbagi'

#python train.py --dataset='MTMdataset' --isCorrupted='corrupted' --module='BNGated'

#python Train.py --dataset='MTMdataset' --isCorrupted='corrupted' --module='OursGated3' --lossBeta=2


#python Train_Combine.py --dataset='Combinedataset' --isCorrupted='uncorrupted' --module='OursGated' --lossBeta=2

#python Train.py --dataset='MTMdataset' --isCorrupted='corrupted' --module='normal' --lossBeta=1

#python Train.py --dataset='MTMdataset' --isCorrupted='corrupted' --module='BNGated' --lossBeta=1.5

# python Train_Combine.py --dataset='Combinedataset' --isCorrupted='corrupted' --module='OursGated' --lossBeta=2

#python Train.py --dataset='MTMdataset' --isCorrupted='uncorrupted' --module='OursGated_soft_test' --lossBeta=1

#python Train_CMU.py --dataset='CMUdataset' --isCorrupted='uncorrupted' --module='OursGated' --lossBeta=2

#python Train_CMU.py --dataset='CMUdataset' --isCorrupted='corrupted' --module='OursGated' --lossBeta=1.5

#python Train_CMU.py --dataset='CMUdataset' --isCorrupted='uncorrupted' --module='skip' --lossBeta=1
#
#python Train_CMU.py --dataset='CMUdataset' --isCorrupted='corrupted' --module='skip' --lossBeta=1

# python Train.py --dataset='MTMdataset' --isCorrupted='uncorrupted' --DEC_M_N_LAYERS=2

# python Train.py --dataset='MTMdataset' --isCorrupted='uncorrupted' --DEC_M_N_LAYERS=4

##### here is the resubmussion

# python Train_CMU.py --dataset='CMUdataset' --isCorrupted='uncorrupted' --module='OursGated' --lossBeta=2 --DEC_M_N_LAYERS=4
# python Train_CMU.py --dataset='CMUdataset' --isCorrupted='uncorrupted' --module='OursGated' --lossBeta=2 --DEC_M_N_LAYERS=8

# python Train_CMU.py --dataset='CMUdataset' --isCorrupted='corrupted' --module='OursGated' --lossBeta=1.5 --DEC_M_N_LAYERS=4
# python Train_CMU.py --dataset='CMUdataset' --isCorrupted='corrupted' --module='OursGated' --lossBeta=1.5 --DEC_M_N_LAYERS=8

# python Train.py --dataset='MTMdataset' --isCorrupted='uncorrupted' --DEC_M_N_LAYERS=8

# python Train.py --dataset='MTMdataset' --isCorrupted='uncorrupted' --module='OursGated' --DEC_M_N_LAYERS=8
# python Train.py --dataset='MTMdataset' --isCorrupted='uncorrupted' --module='OursGated' --DEC_M_N_LAYERS=4

# python Train.py --dataset='MTMdataset' --isCorrupted='corrupted' --module='OursGated' --DEC_M_N_LAYERS=8
# python Train.py --dataset='MTMdataset' --isCorrupted='corrupted' --module='OursGated' --DEC_M_N_LAYERS=4

python Train_Combine.py --dataset='Combinedataset' --isCorrupted='uncorrupted' --module='OursGated' --DEC_M_N_LAYERS=8
python Train_Combine.py --dataset='Combinedataset' --isCorrupted='uncorrupted' --module='OursGated' --DEC_M_N_LAYERS=4

python Train_Combine.py --dataset='Combinedataset' --isCorrupted='corrupted' --module='OursGated' --DEC_M_N_LAYERS=8
python Train_Combine.py --dataset='Combinedataset' --isCorrupted='corrupted' --module='OursGated' --DEC_M_N_LAYERS=4