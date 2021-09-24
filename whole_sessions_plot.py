# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 09:11:04 2021

@author: frede
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#from matplotlib import rcParams
#rcParams['font.family'] = 'Times New Roman' #changing the font

# =============================================================================
# Figure 7 plot. This file is only attached for illustration purposes and the sake of completeness.
# Note: the dictionaries for the bar plots can be extracted from eval_base_study module (call method: prepare_results_for_plot_whole_sessions(100)).
# However, all results are based on randomized training and testing sets; therefore, reproducing results via function call 
# will most likely lead to similar, but not the exact same results.
# When a key had only zero list values for all algorithms, it had been removed from the dictionary.  
# =============================================================================

fig, axs = plt.subplots(1, 3, figsize = (6,5), gridspec_kw={'width_ratios': [4, 4, 1]})#, constrained_layout = True)

ax = axs[0]

##PCA bar plot

lda_pca = {'3':[0,1,0], '4':[0,5,4], '5': [1,2,11], '6': [7,4,20], '7': [13,24,25], '8': [15,14,23], '9': [30,33,10], '10': [24,9,3], '11': [9,7,4], '12': [1,1,0]}
rf_pca = {'3':[3,2,4], '4':[8,4,5],'5': [18,7,14], '6': [22,17,16], '7': [18,30,28], '8': [18,20,24], '9': [8,13,5], '10': [5,7,3], '11': [0,0,1], '12': [0,0,0]}
ff_pca = {'3':[1,0,2], '4':[5,5,3], '5': [12,11,15], '6': [19,16,16], '7': [27,21,26], '8': [16,23,26], '9': [16,18,9], '10': [4,6,1], '11': [0,0,1], '12': [0,0,1]} 
svmrbf_pca = {'3':[2,2,2], '4':[7,3,5],'5': [16,6,13], '6': [23,17,28], '7': [22,32,20], '8':[22,22,20], '9':[6,13,10], '10':[2,4,1], '11':[0,1,1], '12':[0,0,0]}
svmlin_pca = {'3':[0,0,0], '4': [0,0,0],'5': [0,1,6], '6': [0,3,4], '7': [3,4,6], '8': [7,14,11], '9':[26,19,27], '10':[18,34,25], '11':[36,20,12], '12':[10,5,9]}

df_lda_pca = pd.DataFrame(lda_pca)
lda_pca_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(lda_pca['3'], lda_pca['4'], lda_pca['5'], lda_pca['6'],lda_pca['7'], lda_pca['8'], lda_pca['9'],lda_pca['10'], lda_pca['11'], lda_pca['12'])]
lda_pca3= [i/j * 100 for i,j in zip(df_lda_pca['3'], lda_pca_totals)]
lda_pca4 = [i/j * 100 for i,j in zip(df_lda_pca['4'], lda_pca_totals)]
lda_pca5 = [i/j * 100 for i,j in zip(df_lda_pca['5'], lda_pca_totals)]
lda_pca6 = [i/j * 100 for i,j in zip(df_lda_pca['6'], lda_pca_totals)]
lda_pca7 = [i/j * 100 for i,j in zip(df_lda_pca['7'], lda_pca_totals)]
lda_pca8 = [i/j * 100 for i,j in zip(df_lda_pca['8'], lda_pca_totals)]
lda_pca9 = [i/j * 100 for i,j in zip(df_lda_pca['9'], lda_pca_totals)]
lda_pca10 = [i/j * 100 for i,j in zip(df_lda_pca['10'], lda_pca_totals)]
lda_pca11 = [i/j * 100 for i,j in zip(df_lda_pca['11'], lda_pca_totals)]
lda_pca12 = [i/j * 100 for i,j in zip(df_lda_pca['12'], lda_pca_totals)]

df_ff_pca = pd.DataFrame(ff_pca)
ff_pca_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(ff_pca['3'], ff_pca['4'], ff_pca['5'], ff_pca['6'],ff_pca['7'], ff_pca['8'], ff_pca['9'],ff_pca['10'], ff_pca['11'], ff_pca['12'])]
ff_pca3= [i/j * 100 for i,j in zip(df_ff_pca['3'], ff_pca_totals)]
ff_pca4 = [i/j * 100 for i,j in zip(df_ff_pca['4'], ff_pca_totals)]
ff_pca5 = [i/j * 100 for i,j in zip(df_ff_pca['5'], ff_pca_totals)]
ff_pca6 = [i/j * 100 for i,j in zip(df_ff_pca['6'], ff_pca_totals)]
ff_pca7 = [i/j * 100 for i,j in zip(df_ff_pca['7'], ff_pca_totals)]
ff_pca8 = [i/j * 100 for i,j in zip(df_ff_pca['8'], ff_pca_totals)]
ff_pca9 = [i/j * 100 for i,j in zip(df_ff_pca['9'], ff_pca_totals)]
ff_pca10 = [i/j * 100 for i,j in zip(df_ff_pca['10'], ff_pca_totals)]
ff_pca11 = [i/j * 100 for i,j in zip(df_ff_pca['11'], ff_pca_totals)]
ff_pca12 = [i/j * 100 for i,j in zip(df_ff_pca['12'], ff_pca_totals)]

df_rf_pca = pd.DataFrame(rf_pca)
rf_pca_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(rf_pca['3'], rf_pca['4'], rf_pca['5'], rf_pca['6'],rf_pca['7'], rf_pca['8'], rf_pca['9'],rf_pca['10'], rf_pca['11'], rf_pca['12'])]
rf_pca3= [i/j * 100 for i,j in zip(df_rf_pca['3'], rf_pca_totals)]
rf_pca4 = [i/j * 100 for i,j in zip(df_rf_pca['4'], rf_pca_totals)]
rf_pca5 = [i/j * 100 for i,j in zip(df_rf_pca['5'], rf_pca_totals)]
rf_pca6 = [i/j * 100 for i,j in zip(df_rf_pca['6'], rf_pca_totals)]
rf_pca7 = [i/j * 100 for i,j in zip(df_rf_pca['7'], rf_pca_totals)]
rf_pca8 = [i/j * 100 for i,j in zip(df_rf_pca['8'], rf_pca_totals)]
rf_pca9 = [i/j * 100 for i,j in zip(df_rf_pca['9'], rf_pca_totals)]
rf_pca10 = [i/j * 100 for i,j in zip(df_rf_pca['10'], rf_pca_totals)]
rf_pca11 = [i/j * 100 for i,j in zip(df_rf_pca['11'], rf_pca_totals)]
rf_pca12 = [i/j * 100 for i,j in zip(df_rf_pca['12'], rf_pca_totals)]

df_svmrbf_pca = pd.DataFrame(svmrbf_pca)
svmrbf_pca_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(svmrbf_pca['3'], svmrbf_pca['4'], svmrbf_pca['5'], svmrbf_pca['6'],svmrbf_pca['7'], svmrbf_pca['8'], svmrbf_pca['9'],svmrbf_pca['10'], svmrbf_pca['11'], svmrbf_pca['12'])]
svmrbf_pca3= [i/j * 100 for i,j in zip(df_svmrbf_pca['3'], svmrbf_pca_totals)]
svmrbf_pca4 = [i/j * 100 for i,j in zip(df_svmrbf_pca['4'], svmrbf_pca_totals)]
svmrbf_pca5 = [i/j * 100 for i,j in zip(df_svmrbf_pca['5'], svmrbf_pca_totals)]
svmrbf_pca6 = [i/j * 100 for i,j in zip(df_svmrbf_pca['6'], svmrbf_pca_totals)]
svmrbf_pca7 = [i/j * 100 for i,j in zip(df_svmrbf_pca['7'], svmrbf_pca_totals)]
svmrbf_pca8 = [i/j * 100 for i,j in zip(df_svmrbf_pca['8'], svmrbf_pca_totals)]
svmrbf_pca9 = [i/j * 100 for i,j in zip(df_svmrbf_pca['9'], svmrbf_pca_totals)]
svmrbf_pca10 = [i/j * 100 for i,j in zip(df_svmrbf_pca['10'], svmrbf_pca_totals)]
svmrbf_pca11 = [i/j * 100 for i,j in zip(df_svmrbf_pca['11'], svmrbf_pca_totals)]
svmrbf_pca12 = [i/j * 100 for i,j in zip(df_svmrbf_pca['12'], svmrbf_pca_totals)]

df_svmlin_pca = pd.DataFrame(svmlin_pca)
svmlin_pca_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(svmlin_pca['3'], svmlin_pca['4'], svmlin_pca['5'], svmlin_pca['6'],svmlin_pca['7'], svmlin_pca['8'], svmlin_pca['9'],svmlin_pca['10'], svmlin_pca['11'], svmlin_pca['12'])]
svmlin_pca3= [i/j * 100 for i,j in zip(df_svmlin_pca['3'], svmlin_pca_totals)]
svmlin_pca4 = [i/j * 100 for i,j in zip(df_svmlin_pca['4'], svmlin_pca_totals)]
svmlin_pca5 = [i/j * 100 for i,j in zip(df_svmlin_pca['5'], svmlin_pca_totals)]
svmlin_pca6 = [i/j * 100 for i,j in zip(df_svmlin_pca['6'], svmlin_pca_totals)]
svmlin_pca7 = [i/j * 100 for i,j in zip(df_svmlin_pca['7'], svmlin_pca_totals)]
svmlin_pca8 = [i/j * 100 for i,j in zip(df_svmlin_pca['8'], svmlin_pca_totals)]
svmlin_pca9 = [i/j * 100 for i,j in zip(df_svmlin_pca['9'], svmlin_pca_totals)]
svmlin_pca10 = [i/j * 100 for i,j in zip(df_svmlin_pca['10'], svmlin_pca_totals)]
svmlin_pca11 = [i/j * 100 for i,j in zip(df_svmlin_pca['11'], svmlin_pca_totals)]
svmlin_pca12 = [i/j * 100 for i,j in zip(df_svmlin_pca['12'], svmlin_pca_totals)]

barWidth = 0.175
barwidth = 0.175
r1 = np.arange(len(lda_pca['5']))        
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]
r4 = [x + barwidth for x in r3]
r5 = [x + barwidth for x in r4]     

ax.bar(r1, lda_pca3, color='#c7e000', width=barWidth, edgecolor='white', label='3')
ax.bar(r1, lda_pca4, color='#52e000', bottom = lda_pca3, width=barWidth, edgecolor='white', label='4')
ax.bar(r1, lda_pca5, color='#00b22c', bottom = [i+j for i,j in zip(lda_pca3, lda_pca4)], width=barWidth, edgecolor='white', label='5')
ax.bar(r1, lda_pca6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(lda_pca3,lda_pca4, lda_pca5)],width=barWidth, edgecolor='white', label='6')
ax.bar(r1, lda_pca7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(lda_pca3,lda_pca4, lda_pca5, lda_pca6)], width=barWidth, edgecolor='white', label='7')
ax.bar(r1, lda_pca8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(lda_pca3,lda_pca4, lda_pca5, lda_pca6, lda_pca7)], width=barWidth, edgecolor='white', label='8')
ax.bar(r1, lda_pca9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(lda_pca3, lda_pca4, lda_pca5, lda_pca6, lda_pca7, lda_pca8)], width=barWidth, edgecolor='white', label='9')
ax.bar(r1, lda_pca10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(lda_pca3, lda_pca4, lda_pca5, lda_pca6, lda_pca7, lda_pca8, lda_pca9)], width=barWidth, edgecolor='white', label='10')
ax.bar(r1, lda_pca11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(lda_pca3, lda_pca4, lda_pca5, lda_pca6, lda_pca7, lda_pca8, lda_pca9,lda_pca10)], width=barWidth, edgecolor='white', label='11')
ax.bar(r1, lda_pca12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(lda_pca3, lda_pca4, lda_pca5, lda_pca6,lda_pca7, lda_pca8, lda_pca9, lda_pca10, lda_pca11)], width=barWidth, edgecolor='white', label='12')

ax.bar(r2, ff_pca3, color='#c7e000', width=barWidth, edgecolor='white', label='3')
ax.bar(r2, ff_pca4, color='#52e000', bottom = ff_pca3, width=barWidth, edgecolor='white', label='4')
ax.bar(r2, ff_pca5, color='#00b22c', bottom = [i+j for i,j in zip(ff_pca3, ff_pca4)], width=barWidth, edgecolor='white', label='5')
ax.bar(r2, ff_pca6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(ff_pca3,ff_pca4, ff_pca5)],width=barWidth, edgecolor='white', label='6')
ax.bar(r2, ff_pca7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(ff_pca3,ff_pca4, ff_pca5, ff_pca6)], width=barWidth, edgecolor='white', label='7')
ax.bar(r2, ff_pca8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(ff_pca3,ff_pca4, ff_pca5, ff_pca6, ff_pca7)], width=barWidth, edgecolor='white', label='8')
ax.bar(r2, ff_pca9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(ff_pca3, ff_pca4, ff_pca5, ff_pca6, ff_pca7, ff_pca8)], width=barWidth, edgecolor='white', label='9')
ax.bar(r2, ff_pca10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(ff_pca3, ff_pca4, ff_pca5, ff_pca6, ff_pca7, ff_pca8, ff_pca9)], width=barWidth, edgecolor='white', label='10')
ax.bar(r2, ff_pca11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(ff_pca3, ff_pca4, ff_pca5, ff_pca6, ff_pca7, ff_pca8, ff_pca9,ff_pca10)], width=barWidth, edgecolor='white', label='11')
ax.bar(r2, ff_pca12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(ff_pca3, ff_pca4, ff_pca5, ff_pca6,ff_pca7, ff_pca8, ff_pca9, ff_pca10, ff_pca11)], width=barWidth, edgecolor='white', label='12')

ax.bar(r3, rf_pca3, color='#c7e000', width=barWidth, edgecolor='white', label='3')
ax.bar(r3, rf_pca4, color='#52e000', bottom = rf_pca3, width=barWidth, edgecolor='white', label='4')
ax.bar(r3, rf_pca5, color='#00b22c', bottom = [i+j for i,j in zip(rf_pca3, rf_pca4)], width=barWidth, edgecolor='white', label='5')
ax.bar(r3, rf_pca6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(rf_pca3,rf_pca4, rf_pca5)],width=barWidth, edgecolor='white', label='6')
ax.bar(r3, rf_pca7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(rf_pca3,rf_pca4, rf_pca5, rf_pca6)], width=barWidth, edgecolor='white', label='7')
ax.bar(r3, rf_pca8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(rf_pca3,rf_pca4, rf_pca5, rf_pca6, rf_pca7)], width=barWidth, edgecolor='white', label='8')
ax.bar(r3, rf_pca9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(rf_pca3, rf_pca4, rf_pca5, rf_pca6, rf_pca7, rf_pca8)], width=barWidth, edgecolor='white', label='9')
ax.bar(r3, rf_pca10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(rf_pca3, rf_pca4, rf_pca5, rf_pca6, rf_pca7, rf_pca8, rf_pca9)], width=barWidth, edgecolor='white', label='10')
ax.bar(r3, rf_pca11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(rf_pca3, rf_pca4, rf_pca5, rf_pca6, rf_pca7, rf_pca8, rf_pca9,rf_pca10)], width=barWidth, edgecolor='white', label='11')
ax.bar(r3, rf_pca12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(rf_pca3, rf_pca4, rf_pca5, rf_pca6,rf_pca7, rf_pca8, rf_pca9, rf_pca10, rf_pca11)], width=barWidth, edgecolor='white', label='12')

ax.bar(r4, svmrbf_pca3, color='#c7e000', width=barWidth, edgecolor='white', label='3')
ax.bar(r4, svmrbf_pca4, color='#52e000', bottom = svmrbf_pca3, width=barWidth, edgecolor='white', label='4')
ax.bar(r4, svmrbf_pca5, color='#00b22c', bottom = [i+j for i,j in zip(svmrbf_pca3, svmrbf_pca4)], width=barWidth, edgecolor='white', label='5')
ax.bar(r4, svmrbf_pca6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(svmrbf_pca3,svmrbf_pca4, svmrbf_pca5)],width=barWidth, edgecolor='white', label='6')
ax.bar(r4, svmrbf_pca7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(svmrbf_pca3,svmrbf_pca4, svmrbf_pca5, svmrbf_pca6)], width=barWidth, edgecolor='white', label='7')
ax.bar(r4, svmrbf_pca8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(svmrbf_pca3,svmrbf_pca4, svmrbf_pca5, svmrbf_pca6, svmrbf_pca7)], width=barWidth, edgecolor='white', label='8')
ax.bar(r4, svmrbf_pca9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(svmrbf_pca3, svmrbf_pca4, svmrbf_pca5, svmrbf_pca6, svmrbf_pca7, svmrbf_pca8)], width=barWidth, edgecolor='white', label='9')
ax.bar(r4, svmrbf_pca10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(svmrbf_pca3, svmrbf_pca4, svmrbf_pca5, svmrbf_pca6, svmrbf_pca7, svmrbf_pca8, svmrbf_pca9)], width=barWidth, edgecolor='white', label='10')
ax.bar(r4, svmrbf_pca11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(svmrbf_pca3, svmrbf_pca4, svmrbf_pca5, svmrbf_pca6, svmrbf_pca7, svmrbf_pca8, svmrbf_pca9,svmrbf_pca10)], width=barWidth, edgecolor='white', label='11')
ax.bar(r4, svmrbf_pca12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(svmrbf_pca3, svmrbf_pca4, svmrbf_pca5, svmrbf_pca6,svmrbf_pca7, svmrbf_pca8, svmrbf_pca9, svmrbf_pca10, svmrbf_pca11)], width=barWidth, edgecolor='white', label='12')

ax.bar(r5, svmlin_pca3, color='#c7e000', width=barWidth, edgecolor='white', label='3')
ax.bar(r5, svmlin_pca4, color='#52e000', bottom = svmlin_pca3, width=barWidth, edgecolor='white', label='4')
ax.bar(r5, svmlin_pca5, color='#00b22c', bottom = [i+j for i,j in zip(svmlin_pca3, svmlin_pca4)], width=barWidth, edgecolor='white', label='5')
ax.bar(r5, svmlin_pca6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(svmlin_pca3,svmlin_pca4, svmlin_pca5)],width=barWidth, edgecolor='white', label='6')
ax.bar(r5, svmlin_pca7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(svmlin_pca3,svmlin_pca4, svmlin_pca5, svmlin_pca6)], width=barWidth, edgecolor='white', label='7')
ax.bar(r5, svmlin_pca8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(svmlin_pca3,svmlin_pca4, svmlin_pca5, svmlin_pca6, svmlin_pca7)], width=barWidth, edgecolor='white', label='8')
ax.bar(r5, svmlin_pca9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(svmlin_pca3, svmlin_pca4, svmlin_pca5, svmlin_pca6, svmlin_pca7, svmlin_pca8)], width=barWidth, edgecolor='white', label='9')
ax.bar(r5, svmlin_pca10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(svmlin_pca3, svmlin_pca4, svmlin_pca5, svmlin_pca6, svmlin_pca7, svmlin_pca8, svmlin_pca9)], width=barWidth, edgecolor='white', label='10')
ax.bar(r5, svmlin_pca11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(svmlin_pca3, svmlin_pca4, svmlin_pca5, svmlin_pca6, svmlin_pca7, svmlin_pca8, svmlin_pca9,svmlin_pca10)], width=barWidth, edgecolor='white', label='11')
ax.bar(r5, svmlin_pca12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(svmlin_pca3, svmlin_pca4, svmlin_pca5, svmlin_pca6,svmlin_pca7, svmlin_pca8, svmlin_pca9, svmlin_pca10, svmlin_pca11)], width=barWidth, edgecolor='white', label='12')


ax.set_xticks([r + 2*barwidth for r in range(len(lda_pca['5']))])
ax.set_xticklabels(['Left', 'Right', 'All'])
ax.set_xlabel("Principal Component Input", fontweight = 'bold', fontsize = 10)
ax.set_ylim(top = 102)
box = ax.get_position()


##Basic feature plot
ax=axs[1]

lda_bf = {'3':[0,2,10],'4': [5,1,8], '5': [4,3,9], '6': [7,6,26], '7': [24,15,22], '8': [24,27,19], '9': [18,21,4], '10': [12,18,2], '11': [5,5,0], '12': [1,2,0]}
ff_bf = {'3':[0,0,4], '4': [10,1,5],'5': [14,14,17], '6': [66,67,65], '7': [4,15,4], '8': [4,3,3], '9': [2,0,2], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]}
rf_bf = {'3':[0,0,0],'4': [0,0,2],'5': [8,2,5], '6': [16,8,6], '7': [19,13,13], '8': [24,28,18], '9': [22,21,24], '10': [4,18,14], '11': [6,9,14], '12': [1,1,4]}  
svmrbf_bf = {'3':[2,1,2], '4':[1,2,8],'5':[6,8,16], '6':[88,20,64], '7':[2,34,10], '8':[1,25,0], '9':[0,8,0], '10':[0,2,0], '11':[0,0,0], '12':[0,0,0]} 
svmlin_bf = {'3':[0,0,0], '4':[0,0,1], '5':[0,0,1], '6':[8,3,6], '7':[8,17,18], '8':[22,26,34], '9':[27,22,20], '10':[24,30,13], '11':[10,2,6], '12':[1,0,1]}


df_lda_bf = pd.DataFrame(lda_bf)
lda_bf_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(lda_bf['3'], lda_bf['4'], lda_bf['5'], lda_bf['6'],lda_bf['7'], lda_bf['8'], lda_bf['9'],lda_bf['10'], lda_bf['11'], lda_bf['12'])]
lda_bf3= [i/j * 100 for i,j in zip(df_lda_bf['3'], lda_bf_totals)]
lda_bf4 = [i/j * 100 for i,j in zip(df_lda_bf['4'], lda_bf_totals)]
lda_bf5 = [i/j * 100 for i,j in zip(df_lda_bf['5'], lda_bf_totals)]
lda_bf6 = [i/j * 100 for i,j in zip(df_lda_bf['6'], lda_bf_totals)]
lda_bf7 = [i/j * 100 for i,j in zip(df_lda_bf['7'], lda_bf_totals)]
lda_bf8 = [i/j * 100 for i,j in zip(df_lda_bf['8'], lda_bf_totals)]
lda_bf9 = [i/j * 100 for i,j in zip(df_lda_bf['9'], lda_bf_totals)]
lda_bf10 = [i/j * 100 for i,j in zip(df_lda_bf['10'], lda_bf_totals)]
lda_bf11 = [i/j * 100 for i,j in zip(df_lda_bf['11'], lda_bf_totals)]
lda_bf12 = [i/j * 100 for i,j in zip(df_lda_bf['12'], lda_bf_totals)]

df_ff_bf = pd.DataFrame(ff_bf)
ff_bf_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(ff_bf['3'], ff_bf['4'], ff_bf['5'], ff_bf['6'],ff_bf['7'], ff_bf['8'], ff_bf['9'],ff_bf['10'], ff_bf['11'], ff_bf['12'])]
ff_bf3= [i/j * 100 for i,j in zip(df_ff_bf['3'], ff_bf_totals)]
ff_bf4 = [i/j * 100 for i,j in zip(df_ff_bf['4'], ff_bf_totals)]
ff_bf5 = [i/j * 100 for i,j in zip(df_ff_bf['5'], ff_bf_totals)]
ff_bf6 = [i/j * 100 for i,j in zip(df_ff_bf['6'], ff_bf_totals)]
ff_bf7 = [i/j * 100 for i,j in zip(df_ff_bf['7'], ff_bf_totals)]
ff_bf8 = [i/j * 100 for i,j in zip(df_ff_bf['8'], ff_bf_totals)]
ff_bf9 = [i/j * 100 for i,j in zip(df_ff_bf['9'], ff_bf_totals)]
ff_bf10 = [i/j * 100 for i,j in zip(df_ff_bf['10'], ff_bf_totals)]
ff_bf11 = [i/j * 100 for i,j in zip(df_ff_bf['11'], ff_bf_totals)]
ff_bf12 = [i/j * 100 for i,j in zip(df_ff_bf['12'], ff_bf_totals)]

df_rf_bf = pd.DataFrame(rf_bf)
rf_bf_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(rf_bf['3'], rf_bf['4'], rf_bf['5'], rf_bf['6'],rf_bf['7'], rf_bf['8'], rf_bf['9'],rf_bf['10'], rf_bf['11'], rf_bf['12'])]
rf_bf3= [i/j * 100 for i,j in zip(df_rf_bf['3'], rf_bf_totals)]
rf_bf4 = [i/j * 100 for i,j in zip(df_rf_bf['4'], rf_bf_totals)]
rf_bf5 = [i/j * 100 for i,j in zip(df_rf_bf['5'], rf_bf_totals)]
rf_bf6 = [i/j * 100 for i,j in zip(df_rf_bf['6'], rf_bf_totals)]
rf_bf7 = [i/j * 100 for i,j in zip(df_rf_bf['7'], rf_bf_totals)]
rf_bf8 = [i/j * 100 for i,j in zip(df_rf_bf['8'], rf_bf_totals)]
rf_bf9 = [i/j * 100 for i,j in zip(df_rf_bf['9'], rf_bf_totals)]
rf_bf10 = [i/j * 100 for i,j in zip(df_rf_bf['10'], rf_bf_totals)]
rf_bf11 = [i/j * 100 for i,j in zip(df_rf_bf['11'], rf_bf_totals)]
rf_bf12 = [i/j * 100 for i,j in zip(df_rf_bf['12'], rf_bf_totals)]

df_svmrbf_bf = pd.DataFrame(svmrbf_bf)
svmrbf_bf_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(svmrbf_bf['3'], svmrbf_bf['4'], svmrbf_bf['5'], svmrbf_bf['6'],svmrbf_bf['7'], svmrbf_bf['8'], svmrbf_bf['9'],svmrbf_bf['10'], svmrbf_bf['11'], svmrbf_bf['12'])]
svmrbf_bf3= [i/j * 100 for i,j in zip(df_svmrbf_bf['3'], svmrbf_bf_totals)]
svmrbf_bf4 = [i/j * 100 for i,j in zip(df_svmrbf_bf['4'], svmrbf_bf_totals)]
svmrbf_bf5 = [i/j * 100 for i,j in zip(df_svmrbf_bf['5'], svmrbf_bf_totals)]
svmrbf_bf6 = [i/j * 100 for i,j in zip(df_svmrbf_bf['6'], svmrbf_bf_totals)]
svmrbf_bf7 = [i/j * 100 for i,j in zip(df_svmrbf_bf['7'], svmrbf_bf_totals)]
svmrbf_bf8 = [i/j * 100 for i,j in zip(df_svmrbf_bf['8'], svmrbf_bf_totals)]
svmrbf_bf9 = [i/j * 100 for i,j in zip(df_svmrbf_bf['9'], svmrbf_bf_totals)]
svmrbf_bf10 = [i/j * 100 for i,j in zip(df_svmrbf_bf['10'], svmrbf_bf_totals)]
svmrbf_bf11 = [i/j * 100 for i,j in zip(df_svmrbf_bf['11'], svmrbf_bf_totals)]
svmrbf_bf12 = [i/j * 100 for i,j in zip(df_svmrbf_bf['12'], svmrbf_bf_totals)]

df_svmlin_bf = pd.DataFrame(svmlin_bf)
svmlin_bf_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(svmlin_bf['3'], svmlin_bf['4'], svmlin_bf['5'], svmlin_bf['6'],svmlin_bf['7'], svmlin_bf['8'], svmlin_bf['9'],svmlin_bf['10'], svmlin_bf['11'], svmlin_bf['12'])]
svmlin_bf3= [i/j * 100 for i,j in zip(df_svmlin_bf['3'], svmlin_bf_totals)]
svmlin_bf4 = [i/j * 100 for i,j in zip(df_svmlin_bf['4'], svmlin_bf_totals)]
svmlin_bf5 = [i/j * 100 for i,j in zip(df_svmlin_bf['5'], svmlin_bf_totals)]
svmlin_bf6 = [i/j * 100 for i,j in zip(df_svmlin_bf['6'], svmlin_bf_totals)]
svmlin_bf7 = [i/j * 100 for i,j in zip(df_svmlin_bf['7'], svmlin_bf_totals)]
svmlin_bf8 = [i/j * 100 for i,j in zip(df_svmlin_bf['8'], svmlin_bf_totals)]
svmlin_bf9 = [i/j * 100 for i,j in zip(df_svmlin_bf['9'], svmlin_bf_totals)]
svmlin_bf10 = [i/j * 100 for i,j in zip(df_svmlin_bf['10'], svmlin_bf_totals)]
svmlin_bf11 = [i/j * 100 for i,j in zip(df_svmlin_bf['11'], svmlin_bf_totals)]
svmlin_bf12 = [i/j * 100 for i,j in zip(df_svmlin_bf['12'], svmlin_bf_totals)]

barWidth = 0.175
r1 = np.arange(len(lda_bf['5']))        
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]
r4 = [x + barwidth for x in r3]
r5 = [x + barwidth for x in r4]     

ax.bar(r1, lda_bf3, color='#c7e000', width=barWidth, edgecolor='white', label='3')
ax.bar(r1, lda_bf4, color='#52e000', bottom = lda_bf3, width=barWidth, edgecolor='white', label='4')
ax.bar(r1, lda_bf5, color='#00b22c', bottom = [i+j for i,j in zip(lda_bf3, lda_bf4)], width=barWidth, edgecolor='white', label='5')
ax.bar(r1, lda_bf6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(lda_bf3,lda_bf4, lda_bf5)],width=barWidth, edgecolor='white', label='6')
ax.bar(r1, lda_bf7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(lda_bf3,lda_bf4, lda_bf5, lda_bf6)], width=barWidth, edgecolor='white', label='7')
ax.bar(r1, lda_bf8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(lda_bf3,lda_bf4, lda_bf5, lda_bf6, lda_bf7)], width=barWidth, edgecolor='white', label='8')
ax.bar(r1, lda_bf9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(lda_bf3, lda_bf4, lda_bf5, lda_bf6, lda_bf7, lda_bf8)], width=barWidth, edgecolor='white', label='9')
ax.bar(r1, lda_bf10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(lda_bf3, lda_bf4, lda_bf5, lda_bf6, lda_bf7, lda_bf8, lda_bf9)], width=barWidth, edgecolor='white', label='10')
ax.bar(r1, lda_bf11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(lda_bf3, lda_bf4, lda_bf5, lda_bf6, lda_bf7, lda_bf8, lda_bf9,lda_bf10)], width=barWidth, edgecolor='white', label='11')
ax.bar(r1, lda_bf12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(lda_bf3, lda_bf4, lda_bf5, lda_bf6,lda_bf7, lda_bf8, lda_bf9, lda_bf10, lda_bf11)], width=barWidth, edgecolor='white', label='12')

ax.bar(r2, ff_bf3, color='#c7e000', width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_bf4, color='#52e000', bottom = ff_bf3, width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_bf5, color='#00b22c', bottom = [i+j for i,j in zip(ff_bf3, ff_bf4)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_bf6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(ff_bf3,ff_bf4, ff_bf5)],width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_bf7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(ff_bf3,ff_bf4, ff_bf5, ff_bf6)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_bf8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(ff_bf3,ff_bf4, ff_bf5, ff_bf6, ff_bf7)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_bf9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(ff_bf3, ff_bf4, ff_bf5, ff_bf6, ff_bf7, ff_bf8)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_bf10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(ff_bf3, ff_bf4, ff_bf5, ff_bf6, ff_bf7, ff_bf8, ff_bf9)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_bf11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(ff_bf3, ff_bf4, ff_bf5, ff_bf6, ff_bf7, ff_bf8, ff_bf9,ff_bf10)], width=barWidth, edgecolor='white', label='')
ax.bar(r2, ff_bf12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(ff_bf3, ff_bf4, ff_bf5, ff_bf6,ff_bf7, ff_bf8, ff_bf9, ff_bf10, ff_bf11)], width=barWidth, edgecolor='white', label='')

ax.bar(r3, rf_bf3, color='#c7e000', width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_bf4, color='#52e000', bottom = rf_bf3, width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_bf5, color='#00b22c', bottom = [i+j for i,j in zip(rf_bf3, rf_bf4)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_bf6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(rf_bf3,rf_bf4, rf_bf5)],width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_bf7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(rf_bf3,rf_bf4, rf_bf5, rf_bf6)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_bf8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(rf_bf3,rf_bf4, rf_bf5, rf_bf6, rf_bf7)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_bf9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(rf_bf3, rf_bf4, rf_bf5, rf_bf6, rf_bf7, rf_bf8)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_bf10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(rf_bf3, rf_bf4, rf_bf5, rf_bf6, rf_bf7, rf_bf8, rf_bf9)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_bf11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(rf_bf3, rf_bf4, rf_bf5, rf_bf6, rf_bf7, rf_bf8, rf_bf9,rf_bf10)], width=barWidth, edgecolor='white', label='')
ax.bar(r3, rf_bf12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(rf_bf3, rf_bf4, rf_bf5, rf_bf6,rf_bf7, rf_bf8, rf_bf9, rf_bf10, rf_bf11)], width=barWidth, edgecolor='white', label='')

ax.bar(r4, svmrbf_bf3, color='#c7e000', width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrbf_bf4, color='#52e000', bottom = svmrbf_bf3, width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrbf_bf5, color='#00b22c', bottom = [i+j for i,j in zip(svmrbf_bf3, svmrbf_bf4)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrbf_bf6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(svmrbf_bf3,svmrbf_bf4, svmrbf_bf5)],width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrbf_bf7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(svmrbf_bf3,svmrbf_bf4, svmrbf_bf5, svmrbf_bf6)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrbf_bf8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(svmrbf_bf3,svmrbf_bf4, svmrbf_bf5, svmrbf_bf6, svmrbf_bf7)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrbf_bf9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(svmrbf_bf3, svmrbf_bf4, svmrbf_bf5, svmrbf_bf6, svmrbf_bf7, svmrbf_bf8)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrbf_bf10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(svmrbf_bf3, svmrbf_bf4, svmrbf_bf5, svmrbf_bf6, svmrbf_bf7, svmrbf_bf8, svmrbf_bf9)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrbf_bf11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(svmrbf_bf3, svmrbf_bf4, svmrbf_bf5, svmrbf_bf6, svmrbf_bf7, svmrbf_bf8, svmrbf_bf9,svmrbf_bf10)], width=barWidth, edgecolor='white', label='')
ax.bar(r4, svmrbf_bf12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(svmrbf_bf3, svmrbf_bf4, svmrbf_bf5, svmrbf_bf6,svmrbf_bf7, svmrbf_bf8, svmrbf_bf9, svmrbf_bf10, svmrbf_bf11)], width=barWidth, edgecolor='white', label='')

ax.bar(r5, svmlin_bf3, color='#c7e000', width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_bf4, color='#52e000', bottom = svmlin_bf3, width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_bf5, color='#00b22c', bottom = [i+j for i,j in zip(svmlin_bf3, svmlin_bf4)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_bf6, color='#1a9391', bottom = [i +j +k for i,j,k in zip(svmlin_bf3,svmlin_bf4, svmlin_bf5)],width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_bf7, color='#00c4da', bottom = [i +j +k+l for i,j,k,l in zip(svmlin_bf3,svmlin_bf4, svmlin_bf5, svmlin_bf6)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_bf8, color='#4643bb', bottom = [i +j +k+l+m for i,j,k,l,m in zip(svmlin_bf3,svmlin_bf4, svmlin_bf5, svmlin_bf6, svmlin_bf7)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_bf9, color='#610c8c', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(svmlin_bf3, svmlin_bf4, svmlin_bf5, svmlin_bf6, svmlin_bf7, svmlin_bf8)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_bf10, color='#980fb7', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(svmlin_bf3, svmlin_bf4, svmlin_bf5, svmlin_bf6, svmlin_bf7, svmlin_bf8, svmlin_bf9)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_bf11, color='#d223fe', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(svmlin_bf3, svmlin_bf4, svmlin_bf5, svmlin_bf6, svmlin_bf7, svmlin_bf8, svmlin_bf9,svmlin_bf10)], width=barWidth, edgecolor='white', label='')
ax.bar(r5, svmlin_bf12, color='#b30347', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(svmlin_bf3, svmlin_bf4, svmlin_bf5, svmlin_bf6,svmlin_bf7, svmlin_bf8, svmlin_bf9, svmlin_bf10, svmlin_bf11)], width=barWidth, edgecolor='white', label='')


ax.set_xticks([r + 2*barwidth for r in range(len(lda_bf['5']))])
ax.set_xticklabels(['Left', 'Right', 'All'])
ax.set_xlabel("Basic Feature Input", fontweight = 'bold', fontsize = 10)
ax.set_ylim(top = 102)
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelleft=False)

box2 = ax.get_position()


#CNN plot
ax = axs[2]
#cnn_whole = {'3': [2, 2, 0], '4': [1, 3, 4], '5': [6, 7, 7], '6': [76, 68, 76], '7': [9, 10, 9], '8': [4, 8, 2], '9': [2, 2, 2], '10': [0, 0, 0], '11': [0, 0, 0], '12': [0, 0, 0]}
cnn_whole = {'3': [1,2,0], '4': [2,2,3], '5': [5,8,7], '6': [75,74,73], '7': [14,10,9], '8': [2,2,7], '9': [1,2,1], '10': [0, 0, 0], '11': [0, 0, 0], '12': [0, 0, 0]}

cnn= pd.DataFrame(cnn_whole)
cnn_totals = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(cnn_whole['3'], cnn_whole['4'], cnn_whole['5'], cnn_whole['6'],cnn_whole['7'], cnn_whole['8'], cnn_whole['9'],cnn_whole['10'], cnn_whole['11'], cnn_whole['12'])]
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
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelleft=False)



handles, labels = axs[1].get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='upper center', fancybox=True, shadow=False, ncol=5, borderpad = 0.5, handlelength = 1.2, bbox_to_anchor = (0.535, box2.y1 + 0.098))
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
