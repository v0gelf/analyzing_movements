# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#from matplotlib import rcParams
#rcParams['font.family'] = 'Times New Roman' #changing the font

# =============================================================================
# Figure 8 plot. This file is only attached for the sake of completeness.
# Note: the dictionaries for the bar plots can be extracted from eval_base_study module (call method: prepare_results_for_plot_segments(100)).
# However, all results are based on randomized training and testing sets; therefore, reproducing results via function call 
# will most likely lead to similar, but not the exact same results.
# When a key had only zero list values for all algorithms, it had been removed from the dictionary.  
# =============================================================================

fig, axs = plt.subplots(1, 3, figsize = (6,5),  gridspec_kw={'width_ratios': [4, 4, 1]})

ax = axs[0]

##Window Plot
lda_windows = {'50': [63,58,73], '200': [25,36,23], '350': [5,5,4], '500': [6,1,0], '650': [1,0,0]}
ff_windows = {'50': [91,86,94], '200': [9,11,6], '350': [0,2,0], '500': [0,1,0], '650': [0,0,0]}
rf_windows = {'50': [25,1,2], '200': [14,11,4], '350': [17,8,8], '500': [22,16,24], '650': [22,64,63]}
svmrbf_windows = {'50':[40,22,24], '200':[26,29,33], '350':[14,25,17], '500':[12,15,16], '650':[8,9,10]}
svmlin_windows = {'50':[10,26,3], '200':[20,14,15], '350':[18,12,23], '500':[24,23,28], '650':[28,25,21]}
cnn_windows = {'50': [1,3,16], '200': [49,42,44], '350': [27,32,23], '500': [15,20,6], '650': [8,3,11]}

df_lda = pd.DataFrame(lda_windows)
lda_totals = [i+j+k+l+m for i,j,k,l,m in zip(df_lda['50'], df_lda['200'], df_lda['350'], df_lda['500'], df_lda['650'])]
lda50 = [i/j * 100 for i,j in zip(df_lda['50'], lda_totals)]
lda200 = [i/j * 100 for i,j in zip(df_lda['200'], lda_totals)]
lda350 = [i/j * 100 for i,j in zip(df_lda['350'], lda_totals)]
lda500 = [i/j * 100 for i,j in zip(df_lda['500'], lda_totals)]
lda650 = [i/j * 100 for i,j in zip(df_lda['650'], lda_totals)]


df_ff = pd.DataFrame(ff_windows)
ff_totals = [i+j+k+l+m for i,j,k,l,m in zip(df_ff['50'], df_ff['200'], df_ff['350'], df_ff['500'], df_ff['650'])]
ff50 = [i/j * 100 for i,j in zip(df_ff['50'], ff_totals)]
ff200 = [i/j * 100 for i,j in zip(df_ff['200'], ff_totals)]
ff350 = [i/j * 100 for i,j in zip(df_ff['350'], ff_totals)]
ff500 = [i/j * 100 for i,j in zip(df_ff['500'], ff_totals)]
ff650 = [i/j * 100 for i,j in zip(df_ff['650'], ff_totals)]

df_rf = pd.DataFrame(rf_windows)
rf_totals = [i+j+k+l+m for i,j,k,l,m in zip(df_rf['50'], df_rf['200'], df_rf['350'], df_rf['500'], df_rf['650'])]
rf50 = [i/j * 100 for i,j in zip(df_rf['50'], rf_totals)]
rf200 = [i/j * 100 for i,j in zip(df_rf['200'], rf_totals)]
rf350 = [i/j * 100 for i,j in zip(df_rf['350'], rf_totals)]
rf500 = [i/j * 100 for i,j in zip(df_rf['500'], rf_totals)]
rf650 = [i/j * 100 for i,j in zip(df_rf['650'], rf_totals)]

df_svmrbf = pd.DataFrame(svmrbf_windows)
svmrbf_totals = [i+j+k+l+m for i,j,k,l,m in zip(df_svmrbf['50'], df_svmrbf['200'], df_svmrbf['350'], df_svmrbf['500'], df_svmrbf['650'])]
svmrbf50 = [i/j * 100 for i,j in zip(df_svmrbf['50'], svmrbf_totals)]
svmrbf200 = [i/j * 100 for i,j in zip(df_svmrbf['200'], svmrbf_totals)]
svmrbf350 = [i/j * 100 for i,j in zip(df_svmrbf['350'], svmrbf_totals)]
svmrbf500 = [i/j * 100 for i,j in zip(df_svmrbf['500'], svmrbf_totals)]
svmrbf650 = [i/j * 100 for i,j in zip(df_svmrbf['650'], svmrbf_totals)]

df_svmlin = pd.DataFrame(svmlin_windows)
svmlin_totals = [i+j+k+l+m for i,j,k,l,m in zip(df_svmlin['50'], df_svmlin['200'], df_svmlin['350'], df_svmlin['500'], df_svmlin['650'])]
svmlin50 = [i/j * 100 for i,j in zip(df_svmlin['50'], svmlin_totals)]
svmlin200 = [i/j * 100 for i,j in zip(df_svmlin['200'], svmlin_totals)]
svmlin350 = [i/j * 100 for i,j in zip(df_svmlin['350'], svmlin_totals)]
svmlin500 = [i/j * 100 for i,j in zip(df_svmlin['500'], svmlin_totals)]
svmlin650 = [i/j * 100 for i,j in zip(df_svmlin['650'], svmlin_totals)]

df_cnn = pd.DataFrame(cnn_windows)
cnn_totals = [i+j+k+l+m for i,j,k,l,m in zip(df_cnn['50'], df_cnn['200'], df_cnn['350'], df_cnn['500'], df_cnn['650'])]
cnn50 = [i/j * 100 for i,j in zip(df_cnn['50'], cnn_totals)]
cnn200 = [i/j * 100 for i,j in zip(df_cnn['200'], cnn_totals)]
cnn350 = [i/j * 100 for i,j in zip(df_cnn['350'], cnn_totals)]
cnn500 = [i/j * 100 for i,j in zip(df_cnn['500'], cnn_totals)]
cnn650 = [i/j * 100 for i,j in zip(df_cnn['650'], cnn_totals)]


barwidth = 0.15
r1 = np.arange(len(lda_windows['50']))        
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]
r4 = [x + barwidth for x in r3]
r5 = [x + barwidth for x in r4]
r6 = [x + barwidth for x in r5]

ax.bar(r1, lda50, color='#b30347', width=barwidth, edgecolor='white', label='50')
ax.bar(r1, lda200, color='#ff3f00', bottom = lda50, width=barwidth, edgecolor='white', label='200')
ax.bar(r1, lda350, color='#ffd221', bottom = [i+j for i,j in zip(lda50, lda200)], width=barwidth, edgecolor='white', label='350')
ax.bar(r1, lda500, color='#52e000', bottom = [i+j+k for i,j,k in zip(lda50, lda200, lda350)], width=barwidth, edgecolor='white', label='500')
bar1 = ax.bar(r1, lda650, color='#00c4da', bottom = [i+j+k+l for i,j,k,l in zip(lda50, lda200, lda350, lda500)], width=barwidth, edgecolor='white', label='650')

ax.bar(r2, ff50, color='#b30347', width=barwidth, edgecolor='white')
ax.bar(r2, ff200, color='#ff3f00', bottom = ff50, width=barwidth, edgecolor='white')
ax.bar(r2, ff350, color='#ffd221', bottom = [i+j for i,j in zip(ff50, ff200)], width=barwidth, edgecolor='white')
ax.bar(r2, ff500, color='#52e000', bottom = [i+j+k for i,j,k in zip(ff50, ff200, ff350)], width=barwidth, edgecolor='white')
ax.bar(r2, ff650, color='#00c4da', bottom = [i+j+k+l for i,j,k,l in zip(ff50, ff200, ff350, ff500)], width=barwidth, edgecolor='white')
        
ax.bar(r3, rf50, color='#b30347', width=barwidth, edgecolor='white')
ax.bar(r3, rf200, color='#ff3f00', bottom = rf50, width=barwidth, edgecolor='white')
ax.bar(r3, rf350, color='#ffd221', bottom = [i+j for i,j in zip(rf50, rf200)], width=barwidth, edgecolor='white')
ax.bar(r3, rf500, color='#52e000', bottom = [i+j+k for i,j,k in zip(rf50, rf200, rf350)], width=barwidth, edgecolor='white')
ax.bar(r3, rf650, color='#00c4da', bottom = [i+j+k+l for i,j,k,l in zip(rf50, rf200, rf350, rf500)], width=barwidth, edgecolor='white')
        
ax.bar(r4, svmrbf50, color='#b30347', width=barwidth, edgecolor='white')
ax.bar(r4, svmrbf200, color='#ff3f00', bottom = svmrbf50, width=barwidth, edgecolor='white')
ax.bar(r4, svmrbf350, color='#ffd221', bottom = [i+j for i,j in zip(svmrbf50, svmrbf200)], width=barwidth, edgecolor='white')
ax.bar(r4, svmrbf500, color='#52e000', bottom = [i+j+k for i,j,k in zip(svmrbf50, svmrbf200, svmrbf350)], width=barwidth, edgecolor='white')
ax.bar(r4, svmrbf650, color='#00c4da', bottom = [i+j+k+l for i,j,k,l in zip(svmrbf50, svmrbf200, svmrbf350, svmrbf500)], width=barwidth, edgecolor='white')

ax.bar(r5, svmlin50, color='#b30347', width=barwidth, edgecolor='white')
ax.bar(r5, svmlin200, color='#ff3f00', bottom = svmlin50, width=barwidth, edgecolor='white')
ax.bar(r5, svmlin350, color='#ffd221', bottom = [i+j for i,j in zip(svmlin50, svmlin200)], width=barwidth, edgecolor='white')
ax.bar(r5, svmlin500, color='#52e000', bottom = [i+j+k for i,j,k in zip(svmlin50, svmlin200, svmlin350)], width=barwidth, edgecolor='white')
ax.bar(r5, svmlin650, color='#00c4da', bottom = [i+j+k+l for i,j,k,l in zip(svmlin50, svmlin200, svmlin350, svmlin500)], width=barwidth, edgecolor='white')

ax.bar(r6, cnn50, color='#b30347', width=barwidth, edgecolor='white')
ax.bar(r6, cnn200, color='#ff3f00', bottom = cnn50, width=barwidth, edgecolor='white')
ax.bar(r6, cnn350, color='#ffd221', bottom = [i+j for i,j in zip(cnn50, cnn200)], width=barwidth, edgecolor='white')
ax.bar(r6, cnn500, color='#52e000', bottom = [i+j+k for i,j,k in zip(cnn50, cnn200, cnn350)], width=barwidth, edgecolor='white')
ax.bar(r6, cnn650, color='#00c4da', bottom = [i+j+k+l for i,j,k,l in zip(cnn50, cnn200, cnn350, cnn500)], width=barwidth, edgecolor='white')

ax.set_xticks([r + 2*barwidth for r in range(len(lda_windows['50']))])
ax.set_xticklabels(['Left', 'Right', 'All'])
ax.set_xlabel("Window Sizes", fontweight = 'bold', fontsize = 10)
ax.set_ylim(top = 102)

##Basic Feature Segment Plot
ax=axs[1]
#results for fixed segments
#lda_seg = {'3':[0,0,0], '4':[0,0,0],'5': [0,0,0], '6': [0,0,0], '7': [0,1,1], '8': [0,3,1], '9': [19,18,7], '10': [27,26,20], '11': [32,37,33], '12': [22,15,38]}
#ff_seg = {'3':[0,0,0], '4':[0,0,0],'5': [0,0,0], '6': [0,0,0], '7': [0,0,1], '8': [4,7,3], '9': [14,16,13], '10': [27,32,21], '11': [37,30,39], '12': [18,15,23]}
#rf_seg = {'3':[0,0,0], '4':[0,0,0],'5': [1,1,0], '6': [1,1,1], '7': [6,8,4], '8': [21,18,10], '9': [34,24,26], '10': [19,27,25], '11': [15,16,24], '12': [3,5,10]}   
#svmrseg_seg = {'3':[0,0,0], '4':[0,0,0],'5':[1,0,0], '6':[4,2,2], '7':[5,3,6], '8':[14,11,21], '9':[24,28,22], '10':[30,31,24], '11':[18,18,19], '12':[4,7,6]}
#svmlin_seg = {'3':[0,0,0], '4':[0,0,0],'5': [0,0,0], '6':[0,0,0], '7':[0,0,0], '8':[0,0,0], '9':[0,1,0], '10':[1,8,5], '11':[19,36,25], '12':[80,55,70]}

#random segments as in paper
lda_seg = {'3':[0,0,0], '4':[0,0,0],'5': [0,0,0], '6': [1,1,1], '7': [2,4,3], '8': [4,24,11], '9': [16,28,26], '10': [30,24,30], '11': [31,24,20], '12': [16,6,9]}
ff_seg = {'3':[0,0,0], '4':[0,0,0],'5': [0,0,0], '6': [0,0,0], '7': [0,0,2], '8': [3,6,3], '9': [17,17,8], '10': [24,27,29], '11': [32,28,39], '12': [24,22,19]}
rf_seg = {'3':[0,0,0], '4':[0,0,0],'5': [0,0,1], '6': [3,1,1], '7': [12,10,3], '8': [16,12,19], '9': [27,22,24], '10': [23,27,29], '11': [15,24,17], '12': [4,4,6]}   
svmrseg_seg = {'3':[0,0,0], '4':[0,0,0],'5':[0,0,0], '6':[2,1,0], '7':[12,3,9], '8':[13,14,14], '9':[22,28,26], '10':[23,23,25], '11':[22,22,16], '12':[6,8,10]}
svmlin_seg = {'3':[0,0,0], '4':[0,0,0],'5': [0,0,0], '6':[0,0,0], '7':[2,0,1], '8':[4,5,1], '9':[13,14,8], '10':[18,20,16], '11':[36,40,42], '12':[27,21,32]}

df_lda_seg = pd.DataFrame(lda_seg)
lda_seg_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(lda_seg['3'], lda_seg['4'], lda_seg['5'], lda_seg['6'],lda_seg['7'], lda_seg['8'], lda_seg['9'],lda_seg['10'], lda_seg['11'], lda_seg['12'])]
lda_seg3= [i/j * 100 for i,j in zip(df_lda_seg['3'], lda_seg_totals)]
lda_seg4 = [i/j * 100 for i,j in zip(df_lda_seg['4'], lda_seg_totals)]
lda_seg5 = [i/j * 100 for i,j in zip(df_lda_seg['5'], lda_seg_totals)]
lda_seg6 = [i/j * 100 for i,j in zip(df_lda_seg['6'], lda_seg_totals)]
lda_seg7 = [i/j * 100 for i,j in zip(df_lda_seg['7'], lda_seg_totals)]
lda_seg8 = [i/j * 100 for i,j in zip(df_lda_seg['8'], lda_seg_totals)]
lda_seg9 = [i/j * 100 for i,j in zip(df_lda_seg['9'], lda_seg_totals)]
lda_seg10 = [i/j * 100 for i,j in zip(df_lda_seg['10'], lda_seg_totals)]
lda_seg11 = [i/j * 100 for i,j in zip(df_lda_seg['11'], lda_seg_totals)]
lda_seg12 = [i/j * 100 for i,j in zip(df_lda_seg['12'], lda_seg_totals)]

df_ff_seg = pd.DataFrame(ff_seg)
ff_seg_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(ff_seg['3'], ff_seg['4'], ff_seg['5'], ff_seg['6'],ff_seg['7'], ff_seg['8'], ff_seg['9'],ff_seg['10'], ff_seg['11'], ff_seg['12'])]
ff_seg3= [i/j * 100 for i,j in zip(df_ff_seg['3'], ff_seg_totals)]
ff_seg4 = [i/j * 100 for i,j in zip(df_ff_seg['4'], ff_seg_totals)]
ff_seg5 = [i/j * 100 for i,j in zip(df_ff_seg['5'], ff_seg_totals)]
ff_seg6 = [i/j * 100 for i,j in zip(df_ff_seg['6'], ff_seg_totals)]
ff_seg7 = [i/j * 100 for i,j in zip(df_ff_seg['7'], ff_seg_totals)]
ff_seg8 = [i/j * 100 for i,j in zip(df_ff_seg['8'], ff_seg_totals)]
ff_seg9 = [i/j * 100 for i,j in zip(df_ff_seg['9'], ff_seg_totals)]
ff_seg10 = [i/j * 100 for i,j in zip(df_ff_seg['10'], ff_seg_totals)]
ff_seg11 = [i/j * 100 for i,j in zip(df_ff_seg['11'], ff_seg_totals)]
ff_seg12 = [i/j * 100 for i,j in zip(df_ff_seg['12'], ff_seg_totals)]

df_rf_seg = pd.DataFrame(rf_seg)
rf_seg_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(rf_seg['3'], rf_seg['4'], rf_seg['5'], rf_seg['6'],rf_seg['7'], rf_seg['8'], rf_seg['9'],rf_seg['10'], rf_seg['11'], rf_seg['12'])]
rf_seg3= [i/j * 100 for i,j in zip(df_rf_seg['3'], rf_seg_totals)]
rf_seg4 = [i/j * 100 for i,j in zip(df_rf_seg['4'], rf_seg_totals)]
rf_seg5 = [i/j * 100 for i,j in zip(df_rf_seg['5'], rf_seg_totals)]
rf_seg6 = [i/j * 100 for i,j in zip(df_rf_seg['6'], rf_seg_totals)]
rf_seg7 = [i/j * 100 for i,j in zip(df_rf_seg['7'], rf_seg_totals)]
rf_seg8 = [i/j * 100 for i,j in zip(df_rf_seg['8'], rf_seg_totals)]
rf_seg9 = [i/j * 100 for i,j in zip(df_rf_seg['9'], rf_seg_totals)]
rf_seg10 = [i/j * 100 for i,j in zip(df_rf_seg['10'], rf_seg_totals)]
rf_seg11 = [i/j * 100 for i,j in zip(df_rf_seg['11'], rf_seg_totals)]
rf_seg12 = [i/j * 100 for i,j in zip(df_rf_seg['12'], rf_seg_totals)]

df_svmrseg_seg = pd.DataFrame(svmrseg_seg)
svmrseg_seg_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(svmrseg_seg['3'], svmrseg_seg['4'], svmrseg_seg['5'], svmrseg_seg['6'],svmrseg_seg['7'], svmrseg_seg['8'], svmrseg_seg['9'],svmrseg_seg['10'], svmrseg_seg['11'], svmrseg_seg['12'])]
svmrseg_seg3= [i/j * 100 for i,j in zip(df_svmrseg_seg['3'], svmrseg_seg_totals)]
svmrseg_seg4 = [i/j * 100 for i,j in zip(df_svmrseg_seg['4'], svmrseg_seg_totals)]
svmrseg_seg5 = [i/j * 100 for i,j in zip(df_svmrseg_seg['5'], svmrseg_seg_totals)]
svmrseg_seg6 = [i/j * 100 for i,j in zip(df_svmrseg_seg['6'], svmrseg_seg_totals)]
svmrseg_seg7 = [i/j * 100 for i,j in zip(df_svmrseg_seg['7'], svmrseg_seg_totals)]
svmrseg_seg8 = [i/j * 100 for i,j in zip(df_svmrseg_seg['8'], svmrseg_seg_totals)]
svmrseg_seg9 = [i/j * 100 for i,j in zip(df_svmrseg_seg['9'], svmrseg_seg_totals)]
svmrseg_seg10 = [i/j * 100 for i,j in zip(df_svmrseg_seg['10'], svmrseg_seg_totals)]
svmrseg_seg11 = [i/j * 100 for i,j in zip(df_svmrseg_seg['11'], svmrseg_seg_totals)]
svmrseg_seg12 = [i/j * 100 for i,j in zip(df_svmrseg_seg['12'], svmrseg_seg_totals)]

df_svmlin_seg = pd.DataFrame(svmlin_seg)
svmlin_seg_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(svmlin_seg['3'], svmlin_seg['4'], svmlin_seg['5'], svmlin_seg['6'],svmlin_seg['7'], svmlin_seg['8'], svmlin_seg['9'],svmlin_seg['10'], svmlin_seg['11'], svmlin_seg['12'])]
svmlin_seg3= [i/j * 100 for i,j in zip(df_svmlin_seg['3'], svmlin_seg_totals)]
svmlin_seg4 = [i/j * 100 for i,j in zip(df_svmlin_seg['4'], svmlin_seg_totals)]
svmlin_seg5 = [i/j * 100 for i,j in zip(df_svmlin_seg['5'], svmlin_seg_totals)]
svmlin_seg6 = [i/j * 100 for i,j in zip(df_svmlin_seg['6'], svmlin_seg_totals)]
svmlin_seg7 = [i/j * 100 for i,j in zip(df_svmlin_seg['7'], svmlin_seg_totals)]
svmlin_seg8 = [i/j * 100 for i,j in zip(df_svmlin_seg['8'], svmlin_seg_totals)]
svmlin_seg9 = [i/j * 100 for i,j in zip(df_svmlin_seg['9'], svmlin_seg_totals)]
svmlin_seg10 = [i/j * 100 for i,j in zip(df_svmlin_seg['10'], svmlin_seg_totals)]
svmlin_seg11 = [i/j * 100 for i,j in zip(df_svmlin_seg['11'], svmlin_seg_totals)]
svmlin_seg12 = [i/j * 100 for i,j in zip(df_svmlin_seg['12'], svmlin_seg_totals)]

barwidth = 0.18
barWidth = 0.18
r1 = np.arange(len(lda_seg['5']))        
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]
r4 = [x + barwidth for x in r3]
r5 = [x + barwidth for x in r4]     

ax.bar(r1, lda_seg3, color='#c7e000', width=barWidth, edgecolor='white', label='')
ax.bar(r1, lda_seg4, color='#52e000', bottom = lda_seg3, width=barWidth, edgecolor='white', label='')
ax.bar(r1, lda_seg5, color='#00b22c', bottom = [i+j for i,j in zip(lda_seg3, lda_seg4)], width=barWidth, edgecolor='white', label='5')
ax.bar(r1, lda_seg6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(lda_seg3,lda_seg4, lda_seg5)],width=barWidth, edgecolor='white', label='6')
ax.bar(r1, lda_seg7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(lda_seg3,lda_seg4, lda_seg5, lda_seg6)], width=barWidth, edgecolor='white', label='7')
ax.bar(r1, lda_seg8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(lda_seg3,lda_seg4, lda_seg5, lda_seg6, lda_seg7)], width=barWidth, edgecolor='white', label='8')
ax.bar(r1, lda_seg9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(lda_seg3, lda_seg4, lda_seg5, lda_seg6, lda_seg7, lda_seg8)], width=barWidth, edgecolor='white', label='9')
ax.bar(r1, lda_seg10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(lda_seg3, lda_seg4, lda_seg5, lda_seg6, lda_seg7, lda_seg8, lda_seg9)], width=barWidth, edgecolor='white', label='10')
ax.bar(r1, lda_seg11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(lda_seg3, lda_seg4, lda_seg5, lda_seg6, lda_seg7, lda_seg8, lda_seg9,lda_seg10)], width=barWidth, edgecolor='white', label='11')
ax.bar(r1, lda_seg12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(lda_seg3, lda_seg4, lda_seg5, lda_seg6,lda_seg7, lda_seg8, lda_seg9, lda_seg10, lda_seg11)], width=barWidth, edgecolor='white', label='12')

ax.bar(r2, ff_seg3, color='#c7e000', width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_seg4, color='#52e000', bottom = ff_seg3, width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_seg5, color='#00b22c', bottom = [i+j for i,j in zip(ff_seg3, ff_seg4)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_seg6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(ff_seg3,ff_seg4, ff_seg5)],width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_seg7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(ff_seg3,ff_seg4, ff_seg5, ff_seg6)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_seg8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(ff_seg3,ff_seg4, ff_seg5, ff_seg6, ff_seg7)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_seg9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(ff_seg3, ff_seg4, ff_seg5, ff_seg6, ff_seg7, ff_seg8)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_seg10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(ff_seg3, ff_seg4, ff_seg5, ff_seg6, ff_seg7, ff_seg8, ff_seg9)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_seg11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(ff_seg3, ff_seg4, ff_seg5, ff_seg6, ff_seg7, ff_seg8, ff_seg9,ff_seg10)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_seg12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(ff_seg3, ff_seg4, ff_seg5, ff_seg6,ff_seg7, ff_seg8, ff_seg9, ff_seg10, ff_seg11)], width=barWidth, edgecolor='white', label='')

ax.bar(r3, rf_seg3, color='#c7e000', width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_seg4, color='#52e000', bottom = rf_seg3, width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_seg5, color='#00b22c', bottom = [i+j for i,j in zip(rf_seg3, rf_seg4)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_seg6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(rf_seg3,rf_seg4, rf_seg5)],width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_seg7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(rf_seg3,rf_seg4, rf_seg5, rf_seg6)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_seg8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(rf_seg3,rf_seg4, rf_seg5, rf_seg6, rf_seg7)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_seg9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(rf_seg3, rf_seg4, rf_seg5, rf_seg6, rf_seg7, rf_seg8)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_seg10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(rf_seg3, rf_seg4, rf_seg5, rf_seg6, rf_seg7, rf_seg8, rf_seg9)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_seg11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(rf_seg3, rf_seg4, rf_seg5, rf_seg6, rf_seg7, rf_seg8, rf_seg9,rf_seg10)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_seg12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(rf_seg3, rf_seg4, rf_seg5, rf_seg6,rf_seg7, rf_seg8, rf_seg9, rf_seg10, rf_seg11)], width=barWidth, edgecolor='white', label='')

ax.bar(r4, svmrseg_seg3, color='#c7e000', width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrseg_seg4, color='#52e000', bottom = svmrseg_seg3, width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrseg_seg5, color='#00b22c', bottom = [i+j for i,j in zip(svmrseg_seg3, svmrseg_seg4)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrseg_seg6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(svmrseg_seg3,svmrseg_seg4, svmrseg_seg5)],width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrseg_seg7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(svmrseg_seg3,svmrseg_seg4, svmrseg_seg5, svmrseg_seg6)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrseg_seg8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(svmrseg_seg3,svmrseg_seg4, svmrseg_seg5, svmrseg_seg6, svmrseg_seg7)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrseg_seg9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(svmrseg_seg3, svmrseg_seg4, svmrseg_seg5, svmrseg_seg6, svmrseg_seg7, svmrseg_seg8)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrseg_seg10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(svmrseg_seg3, svmrseg_seg4, svmrseg_seg5, svmrseg_seg6, svmrseg_seg7, svmrseg_seg8, svmrseg_seg9)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrseg_seg11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(svmrseg_seg3, svmrseg_seg4, svmrseg_seg5, svmrseg_seg6, svmrseg_seg7, svmrseg_seg8, svmrseg_seg9,svmrseg_seg10)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrseg_seg12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(svmrseg_seg3, svmrseg_seg4, svmrseg_seg5, svmrseg_seg6,svmrseg_seg7, svmrseg_seg8, svmrseg_seg9, svmrseg_seg10, svmrseg_seg11)], width=barWidth, edgecolor='white', label='')

ax.bar(r5, svmlin_seg3, color='#c7e000', width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_seg4, color='#52e000', bottom = svmlin_seg3, width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_seg5, color='#00b22c', bottom = [i+j for i,j in zip(svmlin_seg3, svmlin_seg4)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_seg6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(svmlin_seg3,svmlin_seg4, svmlin_seg5)],width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_seg7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(svmlin_seg3,svmlin_seg4, svmlin_seg5, svmlin_seg6)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_seg8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(svmlin_seg3,svmlin_seg4, svmlin_seg5, svmlin_seg6, svmlin_seg7)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_seg9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(svmlin_seg3, svmlin_seg4, svmlin_seg5, svmlin_seg6, svmlin_seg7, svmlin_seg8)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_seg10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(svmlin_seg3, svmlin_seg4, svmlin_seg5, svmlin_seg6, svmlin_seg7, svmlin_seg8, svmlin_seg9)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_seg11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(svmlin_seg3, svmlin_seg4, svmlin_seg5, svmlin_seg6, svmlin_seg7, svmlin_seg8, svmlin_seg9,svmlin_seg10)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_seg12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(svmlin_seg3, svmlin_seg4, svmlin_seg5, svmlin_seg6,svmlin_seg7, svmlin_seg8, svmlin_seg9, svmlin_seg10, svmlin_seg11)], width=barWidth, edgecolor='white', label='')


ax.set_xticks([r + 2*barwidth for r in range(len(lda_seg['5']))])
ax.set_xticklabels(['Left', 'Right', 'All'])
ax.set_xlabel("Basic Feature Segments", fontweight = 'bold', fontsize = 10)
ax.set_ylim(top = 102)
ax.tick_params(
    axis='y',          
    which='both',      
    bottom=False,      
    top=False,         
    labelleft=False)


box2 = ax.get_position()

#CNN Plot
ax = axs[2]
cnn_seg = {'3':[0,0,0], '4': [0,0,0], '5':[0,0,0], '6':[0,0,0],'7': [0,1,1], '8': [1,3,8], '9': [6, 10,14], '10': [17, 12,31], '11': [30, 47,29], '12': [46, 27,17]}
cnn_seg = {'3':[0,0,0], '4': [0,0,0], '5':[0,0,0], '6':[0,0,0],'7': [3,2,0], '8': [7,7,6], '9': [18,10,15], '10': [17,32,33], '11': [32,33,36], '12': [23,16,10]}
cnn= pd.DataFrame(cnn_seg)
cnn_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(cnn_seg['3'], cnn_seg['4'], cnn_seg['5'], cnn_seg['6'],cnn_seg['7'], cnn_seg['8'], cnn_seg['9'],cnn_seg['10'], cnn_seg['11'], cnn_seg['12'])]
cnn3= [i/j * 100 for i,j in zip(cnn['3'], cnn_totals)]
cnn4 = [i/j * 100 for i,j in zip(cnn['4'], cnn_totals)]
cnn5 = [i/j * 100 for i,j in zip(cnn['5'], cnn_totals)]
cnn6 = [i/j * 100 for i,j in zip(cnn['6'], cnn_totals)]
cnn7 = [i/j * 100 for i,j in zip(cnn['7'], cnn_totals)]
cnn8 = [i/j * 100 for i,j in zip(cnn['8'], cnn_totals)]
cnn9 = [i/j * 100 for i,j in zip(cnn['9'], cnn_totals)]
cnn10 = [i/j * 100 for i,j in zip(cnn['10'], cnn_totals)]
cnn11 = [i/j * 100 for i,j in zip(cnn['11'], cnn_totals)]
cnn12 = [i/j * 100 for i,j in zip(cnn['12'], cnn_totals)]

r = [0,0.45,0.9]
barWidth = 0.3
names = ['L', 'R', 'A']
#ax = axs[0,1]
ax.bar(r, cnn3, color='#c7e000', width=barWidth, edgecolor='white', label='3')
ax.bar(r, cnn4, color='#52e000', bottom = cnn3, width=barWidth, edgecolor='white', label='4')
ax.bar(r, cnn5, color='#00b22c', bottom = [i+j for i,j in zip(cnn3, cnn4)], width=barWidth, edgecolor='white', label='5')
ax.bar(r, cnn6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(cnn3,cnn4, cnn5)],width=barWidth, edgecolor='white', label='6')
ax.bar(r, cnn7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(cnn3,cnn4, cnn5, cnn6)], width=barWidth, edgecolor='white', label='7')
ax.bar(r, cnn8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(cnn3,cnn4, cnn5, cnn6, cnn7)], width=barWidth, edgecolor='white', label='8')
ax.bar(r, cnn9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(cnn3, cnn4, cnn5, cnn6, cnn7, cnn8)], width=barWidth, edgecolor='white', label='9')
ax.bar(r, cnn10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(cnn3, cnn4, cnn5, cnn6, cnn7, cnn8, cnn9)], width=barWidth, edgecolor='white', label='10')
ax.bar(r, cnn11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(cnn3, cnn4, cnn5, cnn6, cnn7, cnn8, cnn9,cnn10)], width=barWidth, edgecolor='white', label='11')
ax.bar(r, cnn12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(cnn3, cnn4, cnn5, cnn6,cnn7, cnn8, cnn9, cnn10, cnn11)], width=barWidth, edgecolor='white', label='12')

ax.set_xticks(r)
ax.set_xticklabels(names)
ax.set_xlabel("CNN", fontweight = 'bold', fontsize = 10)
ax.set_ylim(top = 102)
ax.tick_params(
    axis='y',          
    which='both',      
    bottom=False,      
    top=False,         
    labelleft=False)




handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', fancybox=True, shadow=False, ncol=4, borderpad = 0.5, handlelength = 1.2, columnspacing = 0.2,  bbox_to_anchor = (0.73, box2.y1 + 0.1))

handles2, labels2 = axs[0].get_legend_handles_labels()
fig.legend(handles2, labels2, loc='upper left', fancybox=True, shadow=False, ncol=3, borderpad = 0.5, handlelength = 1.2, columnspacing = 0.2, bbox_to_anchor = (0.1, box2.y1 + 0.1))
# plt.suptitle('Linear Discriminant Analysis & Support Vector Machines',
#               y=0.98, fontsize=15)
fig.tight_layout()
plt.show()

f = plt.gcf()
axes = f.axes
ax = axes[0]
box = ax.get_position()

ax.set_position([box.x0 , box.y0,
                  box.width, box.height - 0.11])

ax1 = axes[1]
box = ax1.get_position()
ax1.set_position([box.x0, box.y0,
                  box.width, box.height - 0.11])

ax2 = axes[2]
box = ax2.get_position()
ax2.set_position([box.x0, box.y0,
                  box.width, box.height - 0.11])
