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


args_opt = parser.parse_args()


# with open('.\\test_results\\result_{}_{}_{}_{}.npy'.format(args_opt.isCorrupted,args_opt.dataset,args_opt.module,args_opt.apikey), 'rb') as f:
#     a = np.load(f)

path = '.\\PSNR_results_2'
text_files = [f for f in os.listdir(path) if f.endswith('.npy')]


results=defaultdict(list)



module_dict={'Baluja': 'Baluja', 'CBAMAttention': 'CBAMGated','normal': 'Gated','relu': 'ReluGated','se': 'Ours','skip': 'SkipGated','OursGated': 'ours(old)','BNGated': 'Ours','OursGated3':'Ours','OursGatedBlock3':'Ours'}

PSNR=[]
Methods=[]
Sigma_o= []
Sigma_o_value=[0.1,0.2,0.3,0.4,0.5]
# oppen all the files and get the results
for i in text_files:
    with open('.\\PSNR_results_2\\{}'.format(i), 'rb') as f:
        a = np.load(f)
        names=i.split('_')
        if names[3]==args_opt.dataset:
            results[module_dict[names[5]]].append(np.round(a,2))


for key, value in results.items():
    print(f"     {key:25s} -> {value}")
    Methods.extend(repeat(key,5))
    PSNR.extend(value)
    Sigma_o.extend(Sigma_o_value)


print(Sigma_o)

plot_dic=defaultdict(list)
plot_dic['Method']=Methods
plot_dic['PSNR (dB)']=PSNR
plot_dic['Sigma_o']=Sigma_o

df = pd.DataFrame.from_dict(plot_dic)


print(df.head())

plt.suptitle("Distorted Signal in {} Dataset".format(args_opt.dataset), fontsize=12, y=0.95)
sns.set_style("darkgrid")
g=sns.lineplot(x='Sigma_o',y='PSNR (dB)',hue='Method',data=df)

# Put the legend out of the figure
g.legend(ncol=3, bbox_to_anchor=(0.5, -0.15), loc='upper center')

plt.tight_layout()
plt.savefig('investigation_result/PSNR_{}_{}_plot.png'.format(args_opt.isCorrupted,args_opt.dataset), dpi=300)

plt.show()








# fig =plt.figure(figsize=(6, 12)) # horizon setting
# gs = gridspec.GridSpec(5, 8) # horizon setting
# plt.subplots_adjust(hspace=0.3)
# plt.suptitle("{} Dataset's MSE Categorized by Parts".format(args_opt.dataset), fontsize=18, y=0.99)
#
# # loop through the length of tickers and keep track of index
# for n, ticker in enumerate(headers):
#
#     # if n<2:
#     #     ax = plt.subplot(gs[0, 4 * n:4 * n + 4])
#     # elif n>=2 and n<4:
#     #     ax = plt.subplot(gs[1, 4 * (n%2):4 * (n%2) + 4])
#     # else:
#     #     ax = plt.subplot(gs[2, 2:6])
#     # add a new subplot iteratively
#     ax = plt.subplot(gs[n, 1:7 ])
#
#     ax.set_title(headers[n])
#
#     # x-axis labels
#     ax.set_xticklabels(modules,fontsize='small')
#
#     # panda operation
#     Group = []
#     corrupted = []
#     uncorrupted = []
#     for i in range(6):
#         Group.extend(repeat(modules[i], data_to_plot.shape[-1]))  # 1026
#         corrupted.extend(data_to_plot[n][i])
#         uncorrupted.extend(data_to_plot[n][i + 6])
#
#     df = pd.DataFrame({'Group': Group, \
#                        'Corrupted': corrupted, 'Uncorrupted': uncorrupted})
#
#     dd = pd.melt(df, id_vars=['Group'], value_vars=['Corrupted', 'Uncorrupted'], var_name='Settings')
#     sns.boxplot(x='Group', y='value', data=dd, hue='Settings', width=0.5, showfliers=False)
#
#
#     # panda operation end
#     # ax.boxplot(data_to_plot[n], flierprops=green_diamond, showfliers=False)
#
#
#     # filter df and plot ticker on the new subplot axis
#     #[df["ticker"] == ticker].plot(ax=ax)
#
#     # chart formatting
#     #ax.set_title(ticker.upper())
#     #ax.get_legend().remove()
#     ax.set_yscale('log')
#     ax.set_xlabel("")
#     ax.grid(color='gray', linestyle='--', linewidth=0.5)
#     if n>0:
#         ax.get_legend().remove()
# #plt.yscale('log')
# plt.tight_layout()
# plt.savefig('investigation_result/{}_{}_plot_ho_v2.png'.format(args_opt.isCorrupted,args_opt.dataset), dpi=300, bbox_inches='tight')

#plt.show()

