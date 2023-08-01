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

#python test.py --dataset='MTM' --isCorrupted='uncorrupted' --module='Baluja' --apikey='2cfbodnn'

#python test.py --dataset='MTM' --isCorrupted='uncorrupted' --module='SEGated' --apikey='1kvh4ffh'

#python test.py --dataset='MTM' --isCorrupted='corrupted' --module='OursGated3' --apikey='2qido2lp'

#python test.py --dataset='MTM' --isCorrupted='corrupted' --module='normal' --apikey='2z91e4v6'

#python Test_CMU.py --dataset='CMU' --isCorrupted='uncorrupted' --module='BNGated' --apikey='2166tr0w'
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='uncorrupted' --module='normal' --apikey='2o79c715'
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='uncorrupted' --module='skip' --apikey='3is4n8fc'
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='uncorrupted' --module='relu' --apikey='t70e57v6'
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='uncorrupted' --module='CBAMAttention' --apikey='377udu4c'
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='uncorrupted' --module='Baluja' --apikey='2gpr66mk'

#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='OursGated3' --apikey='2cilhwbb'

#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='normal' --apikey='3q3wn3xt'
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='skip' --apikey='1x7asybi'
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='relu' --apikey='4i0nd5v6'
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='CBAMAttention' --apikey='12cvqe0n'
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='Baluja' --apikey='2ymajjz0'

#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='normal' --apikey='3q3wn3xt'

#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='skip' --apikey='2bfs6h72'
#
#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='relu' --apikey='1c1gxg71'
#
#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='CBAMAttention' --apikey='2s8dgq7d'
#
#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='Baluja' --apikey='nfhmlgt2'

#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='BNGated' --apikey='14eem6s0'

#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='OursGated3' --apikey='6n6ncapt'

#python Test_Combine.py --dataset='Combinedataset' --isCorrupted='uncorrupted' --module='OursGated3' --apikey='1sgpxg1d'

python Test.py --dataset='MTMdataset' --isCorrupted='corrupted' --module='OursGated3' --apikey='3mannqjn'

#python Test_Combine.py --dataset='Combinedataset' --isCorrupted='corrupted' --module='OursGated3' --apikey='3bucw8m2'