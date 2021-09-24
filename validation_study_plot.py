# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:27:49 2021

@author: frede
"""
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Figure 9 plot. This file is only attached for illustration purposes and the sake of completeness.
# Note: the dictionaries for the bar plots can be extracted from eval_validation_study module (call method: prepare_results_for_plot_validation(100)).
# However, all results are based on randomized segments extracted from (fixed) training and testing sets; therefore, reproducing the exact results as below is still
# rather unlikely. 
# When a key had only zero list values for all algorithms, it had been removed from the dictionary.  
# =============================================================================

#first component LDA, second FF, third RF, fourth SVM (RBF), fifth SVM(LIN), sixth CNN
lieb_200 = {'26':[0,0,2,3,0,0], '27':[0,0,3,12,0,0], '28':[0,3,7,21,0,0], '29':[0,6,18,19,0,0], '30':[0,21,23,23,0,0], '31':[0,20,12,16,0,2], '32':[0,23,16,5,1,4], '33':[1,19,12,1,5,2],
            '34':[9,5,5,0,16,11], '35':[38,3,1,0,27,21], '36':[37,0,1,0,24,22], '37':[10,0,0,0,19,28], '38':[3,0,0,0,8,8], '39':[2,0,0,0,0,2]}

lieb_200 = pd.DataFrame(lieb_200)
lieb_200_totals = [i+j+k+l+m+n+o+p+q+r+s+t+u+v for i,j,k,l,m,n,o,p,q,r,s,t,u,v in zip(lieb_200['26'], lieb_200['27'], lieb_200['28'], lieb_200['29'],lieb_200['30'], lieb_200['31'], lieb_200['32'],lieb_200['33'], lieb_200['34'], lieb_200['35'], lieb_200['36'], lieb_200['37'], lieb_200['38'], lieb_200['39'])]
lieb26= [i/j * 100 for i,j in zip(lieb_200['26'], lieb_200_totals)]
lieb27 = [i/j * 100 for i,j in zip(lieb_200['27'], lieb_200_totals)]
lieb28 = [i/j * 100 for i,j in zip(lieb_200['28'], lieb_200_totals)]
lieb29 = [i/j * 100 for i,j in zip(lieb_200['29'], lieb_200_totals)]
lieb30 = [i/j * 100 for i,j in zip(lieb_200['30'], lieb_200_totals)]
lieb31 = [i/j * 100 for i,j in zip(lieb_200['31'], lieb_200_totals)]
lieb32 = [i/j * 100 for i,j in zip(lieb_200['32'], lieb_200_totals)]
lieb33 = [i/j * 100 for i,j in zip(lieb_200['33'], lieb_200_totals)]
lieb34 = [i/j * 100 for i,j in zip(lieb_200['34'], lieb_200_totals)]
lieb35 = [i/j * 100 for i,j in zip(lieb_200['35'], lieb_200_totals)]
lieb36 = [i/j * 100 for i,j in zip(lieb_200['36'], lieb_200_totals)]
lieb37 = [i/j * 100 for i,j in zip(lieb_200['37'], lieb_200_totals)]
lieb38 = [i/j * 100 for i,j in zip(lieb_200['38'], lieb_200_totals)]
lieb39 = [i/j * 100 for i,j in zip(lieb_200['39'], lieb_200_totals)]

r = [0,1,2,3,4,5]
barWidth = 0.6
names = ['LDA', 'FF', 'RF','SVM(RBF)', 'SVM(LIN)', 'CNN']
fig, ax = plt.subplots(figsize=(6,4))

ax.bar(r, lieb26, color='#c2755e', width=barWidth, edgecolor='white', label='26')
ax.bar(r, lieb27, color='#c2945e', bottom = lieb26, width=barWidth, edgecolor='white', label='27')
ax.bar(r, lieb28, color='#c2ad5e', bottom = [i+j for i,j in zip(lieb26, lieb27)], width=barWidth, edgecolor='white', label='28')
ax.bar(r, lieb29, color='#dada59', bottom = [i +j +k for i,j,k in zip(lieb26,lieb27, lieb28)],width=barWidth, edgecolor='white', label='29')
ax.bar(r, lieb30, color='#abcd46', bottom = [i +j +k+l for i,j,k,l in zip(lieb26,lieb27, lieb28, lieb29)], width=barWidth, edgecolor='white', label='30')
ax.bar(r, lieb31, color='#6fcd46', bottom = [i +j +k+l+m for i,j,k,l,m in zip(lieb26,lieb27, lieb28, lieb29, lieb30)], width=barWidth, edgecolor='white', label='31')
ax.bar(r, lieb32, color='#46cd83', bottom = [i+j+k+l+m+n for i,j,k,l,m,n in zip(lieb26, lieb27, lieb28, lieb29, lieb30, lieb31)], width=barWidth, edgecolor='white', label='32')
ax.bar(r, lieb33, color='#46cdb2', bottom = [i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(lieb26, lieb27, lieb28, lieb29, lieb30, lieb31, lieb32)], width=barWidth, edgecolor='white', label='33')
ax.bar(r, lieb34, color='#37bdce', bottom = [i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(lieb26, lieb27, lieb28, lieb29, lieb30, lieb31, lieb32, lieb33)], width=barWidth, edgecolor='white', label='34')
ax.bar(r, lieb35, color='#3798ce', bottom = [i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(lieb26, lieb27, lieb28, lieb29, lieb30, lieb31, lieb32, lieb33, lieb34)], width=barWidth, edgecolor='white', label='35')
ax.bar(r, lieb36, color='#375ece', bottom = [i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(lieb26, lieb27, lieb28, lieb29, lieb30, lieb31, lieb32, lieb33, lieb34, lieb35)], width=barWidth, edgecolor='white', label='36')
ax.bar(r, lieb37, color='#5a37ce', bottom = [i+j+k+l+m+n+o+p+q+r+s for i,j,k,l,m,n,o,p,q,r,s in zip(lieb26, lieb27, lieb28, lieb29, lieb30, lieb31, lieb32, lieb33, lieb34, lieb35, lieb36)], width=barWidth, edgecolor='white', label='37')
ax.bar(r, lieb38, color='#8e37ce', bottom = [i+j+k+l+m+n+o+p+q+r+s+t for i,j,k,l,m,n,o,p,q,r,s,t in zip(lieb26, lieb27, lieb28, lieb29, lieb30, lieb31, lieb32, lieb33, lieb34, lieb35, lieb36, lieb37)], width=barWidth, edgecolor='white', label='38')
ax.bar(r, lieb39, color='#55117a', bottom = [i+j+k+l+m+n+o+p+q+r+s+t+u for i,j,k,l,m,n,o,p,q,r,s,t,u in zip(lieb26, lieb27, lieb28, lieb29, lieb30, lieb31, lieb32, lieb33, lieb34, lieb35, lieb36, lieb37, lieb38)], width=barWidth, edgecolor='white', label='39')

ax.set_xticks(r)
ax.set_xticklabels(names)#, fontsize = 10)#,fontsize = 12)
#ax.set_xlabel("Correct Predictions on Validation Data", fontweight = 'bold', fontsize = 10)
ax.set_ylim(top = 102)

ax = plt.gca()
box = ax.get_position()
#ax.set_position([box.x0, box.y0,
#                 box.width, box.height])

ax.legend(loc='upper center',
          fancybox=True, shadow=False, ncol=7, columnspacing = 0.2, handlelength = 1.2, bbox_to_anchor=(0.5, 1.195))

fig.tight_layout()
plt.show()

f = plt.gcf()
axes = f.axes
ax = axes[0]
box = ax.get_position()

ax.set_position([box.x0 , box.y0,
                  box.width, box.height])
    
f.tight_layout()