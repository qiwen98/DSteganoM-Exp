#!/bin/bash

set -e


# normal, relu, skip, Baluja, CBAMAttention, OursGated




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


for value in $(seq 0.1 0.1 0.5)
do
echo $value
#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='normal' --apikey='2z91e4v6' --sigma_o=$value

#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='skip' --apikey='2bfs6h72' --sigma_o=$value
#
#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='relu' --apikey='1c1gxg71' --sigma_o=$value
#
#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='CBAMAttention' --apikey='2s8dgq7d' --sigma_o=$value
#
#python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='Baluja' --apikey='nfhmlgt2' --sigma_o=$value

python Test.py --dataset='MTM' --isCorrupted='corrupted' --module='OursGated3' --apikey='3mannqjn' --sigma_o=$value

#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='BNGated' --apikey='14eem6s0' --sigma_o=$value
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='normal' --apikey='3q3wn3xt' --sigma_o=$value
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='skip' --apikey='1x7asybi' --sigma_o=$value
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='relu' --apikey='4i0nd5v6' --sigma_o=$value
#
#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='CBAMAttention' --apikey='12cvqe0n' --sigma_o=$value

#python Test_CMU.py --dataset='CMU' --isCorrupted='corrupted' --module='Baluja' --apikey='2ymajjz0' --sigma_o=$value

done

