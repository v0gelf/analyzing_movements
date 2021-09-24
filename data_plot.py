# -*- coding: utf-8 -*-


import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from eval_base_study import random_signal, standardize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from matplotlib import rcParams
from sklearn import tree
import pydotplus
import os

#font settings
#from matplotlib import rc
#rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Times New Roman'
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

# =============================================================================
# Plot helper functions: See section below to directly create the plots displayed in the manuscript
# =============================================================================

#path to locate prepped data
dirname = os.path.dirname(__file__)
train_path = os.path.join(dirname, 'traindata/').replace("\\","/")

#path to save tree plot
tree_path = train_path + "ex_tree.pdf"

#helper function for svm plot extracted from the Python Data Science Handbook by Jake VanderPlas
#URL: https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


#plot first principal component for selected body part and channel (feature) separated for groups 0 and 1 (control vs experimental)
def plot_pca(body_part, feature, boxplot = False):
    """
    body_part : either "Right_Hand" or "Left_Hand"
    feature : either "x-Position", "y-Position" or "z-Position" for global/absolute positioning;
                "x-Position (rel)", "y-Position (rel)" or "z-Position (rel)" for local/relative positioning 
    boxplot : plot additional bloxplot of pc data

    """
    try:
        pickle_in = open(train_path + body_part + ".pickle", "rb")
        df = pickle.load(pickle_in)
    except: 
        print("Invalid Body Part entered. Must be 'Left_Hand' or 'Right_Hand'")
    
    df = standardize(df, [i for i in range(72)])
    df_res, targets = random_signal(1, df, window_size = 4970)
    options = {"x-Position": 0, "y-Position": 1, "z-Position": 2, "x-Position (rel)": 3, "y-Position (rel)": 4, "z-Position (rel)": 5}
    df_channel = df_res[:,:,options[feature]]
    
    pca = PCA(n_components = 4)
    df_transform = pca.fit_transform(df_channel)
    
    pca_data = pd.DataFrame(data = df_transform, columns = ['PC1', 'PC2', 'PC3', 'PC4'])
    pca_data['Target'] = [str(int(t)) for t in targets]
    
    if boxplot:
        group0 = pca_data.iloc[np.where(pca_data['Target'] == "0")[0],0]
        group1 = pca_data.iloc[np.where(pca_data['Target'] == "1")[0],0]
        fig1, ax1 = plt.subplots(2,1, figsize = (14,4))
        bplot = ax1[1].boxplot([group0, group1], labels = ['Target 0', 'Target 1'], patch_artist = True, vert = False, positions = [1,1.5], widths = 0.35)
    
        colors = ['red', 'steelblue']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)        

        ax1[1].set_aspect("auto")
        ax = ax1[0]
    else:
        fig, ax = plt.subplots()
    
    sns.kdeplot(data = pca_data, x = "PC1", hue = "Target", palette = ["red", "steelblue"], ax = ax).set_title("" + feature + " (" + '{:2.1%}'.format(pca.explained_variance_ratio_[0]) + " explained variance)", fontsize = "11")
   
    #change tick settings (remove y ticks)
    plt.tick_params(
    axis='y',          
    which='both',      
    bottom=False,     
    top=False,         
    labelleft=False) 
    
    ax.tick_params(
    axis='y',          
    which='both',      
    bottom=False,      
    top=False,         
    labelleft=False) 
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    #move y label
    ax = plt.gca()
    #ax.yaxis.set_label_coords(-0.03,0.5)
    plt.tight_layout()
    
    
#f = plt.gcf()
#f.set_size_inches(6,3)
    


#2D SVM Mean-Plot based on absolute and relative y-coordinate
def plot_svm(body_part):
    """
    body_part : Either "Right_Hand" or "Left_Hand"

    """
    try:
        pickle_in = open(train_path + body_part + ".pickle", "rb")
        df = pickle.load(pickle_in)
    except: 
        print("Invalid Body Part entered. Must be 'Left_Hand' or 'Right_Hand'")
    
    df = standardize(df, [i for i in range(72)])
    df = df[:,[0,1,3,6,8]] #only y-coordinates (may be changed via param input)
    df_res, targets = random_signal(1, df, window_size = 4970)
    
    means = np.mean(df_res, axis = 1)
    clf = svm.SVC(kernel = "linear")
    _ = clf.fit(means, targets)
    
    pd_df = pd.DataFrame({'Mean y-Position (absolute)': means[:,0], 'Mean y-Position (relative)': means[:,1], 'Target': ["0"  if i == 0 else "1" for i in targets]})
    fig, ax = plt.subplots(figsize = (5,3.3))
    sns.scatterplot(x = 'Mean y-Position (absolute)', y = 'Mean y-Position (relative)', hue = 'Target', data = pd_df, palette = ["red", "steelblue"], ax = ax)
    plot_svc_decision_function(clf)
    ax.set_xlabel('Mean y-Position (absolute)', fontsize = 11)
    ax.set_ylabel('Mean y-Position (relative)', fontsize = 11)
    plt.tight_layout()
    


#f = plt.gcf() 
#f.set_size_inches(6,4.5)   #NOT USED
    

#plot example in three axes (absolute/global values)    
def plot_example(body_part):
    """
    body_part : Either "Right_Hand" or "Left_Hand"

    """
    try:
        pickle_in = open(train_path + body_part + ".pickle", "rb")
        df = pickle.load(pickle_in)
    except: 
        print("Invalid Body Part entered. Must be 'Left_Hand' or 'Right_Hand'")
    
    df = standardize(df, [i for i in range(72)])
    df_res, targets = random_signal(1, df, window_size = 4970)
    
    x_pos = df_res[0,:,2]
    y_pos = df_res[0,:,3]
    z_pos = df_res[0,:,4]
    
    fig, ax = plt.subplots(1,3, figsize = (14,3))
    ax[0].plot(x_pos, color = "k")
    ax[1].plot(y_pos, color = "k")
    ax[2].plot(z_pos, color = "k")

   
    ax[0].set_title("x-Position", fontsize = 11)
    ax[1].set_title("y-Position", fontsize = 11)
    ax[2].set_title("z-Position", fontsize = 11)
    
    #remove axes labels for second and third subplot
    ax[1].tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      
    bottom=False,      
    top=False,         
    labelleft=False) 
    
    ax[2].tick_params(
    axis='y',          
    which='both',      
    bottom=False,      
    top=False,         
    labelleft=False) 
    
    
    y_min, y_max = [min([ax[0].get_ylim()[0], ax[1].get_ylim()[0], ax[2].get_ylim()[0]]), max([ax[0].get_ylim()[1], ax[1].get_ylim()[1], ax[2].get_ylim()[1]])]
    for i in range(3):
        ax[i].set_ylim((y_min, y_max))
        
    plt.tight_layout()

    # = ax.get_position()
    
#f = plt.gcf()
#f.set_size_inches(6,2) #funktioniert so besser, als wenn direkt figsize = (6,2)

#plot example tree and save to disc
def plot_tree(body_part):    
    try:
        pickle_in = open(train_path + body_part + ".pickle", "rb")
        df = pickle.load(pickle_in)
    except: 
        print("Invalid Body Part entered. Must be 'Head', 'Left' or 'Right'")
    
    df = standardize(df, [i for i in range(72)])
    df_res, targets = random_signal(1, df, window_size = 4970)
    
    means = np.mean(df_res, axis = 1)
    maxs = np.max(df_res, axis = 1)
    df_bf = np.concatenate((means, maxs), axis = 1)
    clf = tree.DecisionTreeClassifier(random_state=0)
    _ = clf.fit(df_bf, targets)
    
    feature_names = ["x-Mean (abs)", "y-Mean (abs)", "z-Mean (abs)", "x-Mean (rel)", "y-Mean (rel)", "z-Mean (rel)",
                      "x-Max (abs)", "y-Max (abs)", "z-Max (abs)", "x-Max (rel)", "y-Max (rel)", "z-Max (rel)"]
    class_names = ["0", "1"]
    dot_data = tree.export_graphviz(clf, out_file = None, filled = True, impurity = False, rounded = True, feature_names= feature_names, max_depth = 3, class_names = class_names)

    graph = pydotplus.graph_from_dot_data(dot_data)
    nodes = graph.get_node_list()
    
    colors =  ('red', 'steelblue', 'white')
    
    for node in nodes:
        if node.get_name() not in ('node', 'edge'):
            #node.set_fontname("DejaVu Sans")
            node.set_fontname("Times-Roman")
            values = clf.tree_.value[int(node.get_name())][0]
            #color only nodes where only one class is present
            if max(values) == sum(values):    
                node.set_fillcolor(colors[np.argmax(values)])
                #mixed nodes get the default color
            else:
                node.set_fillcolor(colors[-1]) #if not a leaf, color in white
        
    graph
    graph.write_pdf(tree_path)
    
    
    
# =============================================================================
# Plots as in the Manuscript
# =============================================================================
    
#Figure 1: Example Plot
def figure_1():
    plot_example("Right_Hand") 
    f = plt.gcf()
    f.set_size_inches(6.2,2) #adjust figure size

#Figure 2: PC and boxplot    
def figure_2():
    plot_pca("Right_Hand", "x-Position (rel)", boxplot = True)
    f = plt.gcf()
    f.set_size_inches(6,3)

#Note: this plot uses graphviz and is directly saved to the disc    
def figure_4():
    plot_tree("Right_Hand")

#Figure 10: SVM decision boundary    
def figure_10():
    plot_svm("Right_Hand")

    


    
    