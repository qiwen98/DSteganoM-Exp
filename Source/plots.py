import numpy as np


import argparse
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from collections import defaultdict
import seaborn as sns
from itertools import repeat
import pandas as pd


parser = argparse.ArgumentParser(description="testing and prepare npy for plot")
parser.add_argument("--dataset", type=str, default="MTM", help="Dataset, default is MTM.")
parser.add_argument("--isCorrupted", type=str, default="uncorrupted", help="corrupted or not, uncorrupted/corrupted")
parser.add_argument("--module", type=str, default="normal", help="Dataset, default is normal")
parser.add_argument("--apikey", type=str, default="21lzg3cz", help="wandb api key !!! important")

args_opt = parser.parse_args()


# with open('.\\test_results\\result_{}_{}_{}_{}.npy'.format(args_opt.isCorrupted,args_opt.dataset,args_opt.module,args_opt.apikey), 'rb') as f:
#     a = np.load(f)

path = '.\\test_results'
text_files = [f for f in os.listdir(path) if f.endswith('.npy')]

#print(text_files)
results=[]
modules=[]
corrupt_state=0

module_dict={'Baluja': 'Baluja', 'CBAMAttention': 'CBAMGated','normal': 'Gated','relu': 'ReluGated','se': 'Ours','skip': 'SkipGated','Ours': 'ours(old)','BNGated': 'Ours','OursGated3':'Ours','OursGatedBlock3':'Ours'}

# oppen all the files and get the results
for i in text_files:
    with open('.\\test_results\\{}'.format(i), 'rb') as f:
        a = np.load(f)
        names=i.split('_')
        if names[2]==args_opt.dataset:
            results.append(a)

            modules.append(module_dict[names[4]])
            print(module_dict[names[4]])
            #corrupt_state=names[1]

print(modules[:6])
print(modules[6:])

data_to_plot=np.stack( results, axis=0 ) # 12, Number of data ,5

data_to_plot=data_to_plot.transpose(2,0,1) # 5, 12, Number of data

print(data_to_plot.shape) # 5,12,NUmber of data

# https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
# corrupted then uncorrupted
# Group=[]
# corrupted=[]
# uncorrupted=[]
# for i in range(6):
#     Group.extend(repeat(modules[i],data_to_plot.shape[-1])) # 1026
#     corrupted.extend(data_to_plot[0][i])
#     uncorrupted.extend(data_to_plot[0][i+6])
#
# print(len(Group))
# print(len(corrupted))
# print(len(uncorrupted))
#
# df = pd.DataFrame({'Group':Group,\
#                   'Corrupted':corrupted,'Uncorrupted':uncorrupted})
#
# dd=pd.melt(df,id_vars=['Group'],value_vars=['Corrupted','Uncorrupted'],var_name='Setting')
# sns.boxplot(x='Group',y='value',data=dd,hue='Setting', width=0.3,showfliers=False)


#data_to_plot=data_to_plot.transpose(1,0)
# data_to_plot=list(data_to_plot) #turn the axis 0 to list
# data_to_plot=[list(i) for i in data_to_plot] #turn the axis 1 to list

if args_opt.dataset=="MTM":
    headers=['(a) Thumb', '(b) Index', '(c) Middle', '(d) Ring', '(e) Pinky']
else:
    headers = ['(a) Hips, Left Leg', '(b) Right Leg', '(c) Upper Body, Head', '(d) Left Shoulder', '(e) Right Shoulder']
## plots
green_diamond = dict(markerfacecolor='g', marker='D')
# fig3, ax1 = plt.subplots()
# ax1.set_title(headers[0])
#
# # x-axis labels
# ax1.set_xticklabels(modules)
#
# ax1.boxplot(data_to_plot[0], flierprops=green_diamond,showfliers=False)
# #
# # ax3.set_title(modules[1])
# #
# # # x-axis labels
# # ax3.set_xticklabels(headers)
# #
# # ax3.boxplot(data_to_plot, flierprops=green_diamond,showfliers=False)
# ax1.set_yscale('log')
# show plot
#https://engineeringfordatascience.com/posts/matplotlib_subplots/

# #calculate interquartile range (IQR)
# stat=data_to_plot[0][3] # ours
#
# q3,q2, q1 = np.percentile(stat, [75, 50,25])
# iqr = q3 - q1
#
# print(q1,q2,q3,iqr)
#
# #calculate interquartile range (IQR)
# stat=data_to_plot[0][5] # skip
#
# q3,q2, q1 = np.percentile(stat, [75, 50,25])
# iqr = q3 - q1
#
# print(q1,q2,q3,iqr)




# fig =plt.figure(figsize=(10, 12))
# gs = gridspec.GridSpec(3, 8)

fig =plt.figure(figsize=(6, 12)) # horizon setting
gs = gridspec.GridSpec(5, 8) # horizon setting
plt.subplots_adjust(hspace=0.3)
plt.suptitle("{} Dataset's MSE Categorized by Parts".format(args_opt.dataset), fontsize=18, y=0.99)

# loop through the length of tickers and keep track of index
for n, ticker in enumerate(headers):

    # if n<2:
    #     ax = plt.subplot(gs[0, 4 * n:4 * n + 4])
    # elif n>=2 and n<4:
    #     ax = plt.subplot(gs[1, 4 * (n%2):4 * (n%2) + 4])
    # else:
    #     ax = plt.subplot(gs[2, 2:6])
    # add a new subplot iteratively
    ax = plt.subplot(gs[n, 1:7 ])

    ax.set_title(headers[n])

    # x-axis labels
    ax.set_xticklabels(modules,fontsize='small')

    # panda operation
    Group = []
    corrupted = []
    uncorrupted = []
    for i in range(6):
        Group.extend(repeat(modules[i], data_to_plot.shape[-1]))  # 1026
        corrupted.extend(data_to_plot[n][i])
        uncorrupted.extend(data_to_plot[n][i + 6])

    df = pd.DataFrame({'Group': Group, \
                       'Corrupted': corrupted, 'Uncorrupted': uncorrupted})

    dd = pd.melt(df, id_vars=['Group'], value_vars=['Corrupted', 'Uncorrupted'], var_name='Settings')
    sns.boxplot(x='Group', y='value', data=dd, hue='Settings', width=0.5, showfliers=False)


    # panda operation end
    # ax.boxplot(data_to_plot[n], flierprops=green_diamond, showfliers=False)


    # filter df and plot ticker on the new subplot axis
    #[df["ticker"] == ticker].plot(ax=ax)

    # chart formatting
    #ax.set_title(ticker.upper())
    #ax.get_legend().remove()
    ax.set_yscale('log')
    ax.set_xlabel("")
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    if n>0:
        ax.get_legend().remove()
#plt.yscale('log')
plt.tight_layout()
plt.savefig('investigation_result/{}_{}_plot_ho_v2.png'.format(args_opt.isCorrupted,args_opt.dataset), dpi=300, bbox_inches='tight')

#plt.show()

