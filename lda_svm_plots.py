# -*- coding: utf-8 -*-

"""
This is a modification/extension of examples available on the Scikit website.
URL: https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html (for LDA)
URL: https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html (for SVM)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

#from matplotlib import rcParams
#rcParams['font.family'] = 'Times New Roman' #changing the font
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = 'Arial'

# #############################################################################
# Colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)


# #############################################################################
# Generate datasets
def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 50, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    #C = np.array([[1, 0.], [1, 0.]])
    #X = np.r_[np.dot(np.random.randn(n, dim), C),
    #          np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y



# #############################################################################
# Plot functions
def plot_data(lda, X, y, y_pred, fig_index):
    fig, splots = plt.subplots(1, 2, figsize = (11, 4.5))#figsize = (11,4.5))
    ax = splots[0]
    ax.set_title("Linear Discriminant Analysis", fontsize = 11)


    tp = (y == y_pred)  # True Positive
    tp0 = tp[y == 0]
    X0 = X[y == 0]
    X0_fp = X0[~tp0]
    #X1_tp, X1_fp = X1[tp1], X1[~tp1]

    #scatter classes
    ax.scatter(X[:, 0], X[:, 1], c= ["red" if y[i] == 0 else "steelblue" for i in range(len(y))], s = 10)# c=y, s=30, cmap=plt.cm.Paired)
    #mark an example false positive
    ax.scatter(X0_fp[1, 0], X0_fp[1,1], s = 40, linewidth = 1, facecolors = 'none', edgecolors = "k", alpha = 0.8)
    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    # plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
    #                norm=colors.Normalize(0., 1.), zorder=0)
    cs = ax.contour(xx, yy, Z, [0.5], linewidths=1, colors='k', alpha = 0.7)

    # means
    ax.plot(lda.means_[0][0], lda.means_[0][1],
             '*', color='yellow', markersize=10, markeredgecolor='grey')
    ax.plot(lda.means_[1][0], lda.means_[1][1],
             '*', color='yellow', markersize=10, markeredgecolor='grey')
    
    
    
    #access two points on the seperating line, to extract slope and intercept and infer perpendicular line
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    v1 = v[:,0]
    v2 = v[:,1]
    line_point1 = [v1[0], v2[0]]
    line_point2 = [v1[10], v2[10]]
    
    #mean coordinates 
    mean_point1 = [lda.means_[0,0], lda.means_[0][1]]
    mean_point2 = [lda.means_[1,0], lda.means_[1][1]]
    
    #midpoint between means as access point passed by the perpendicular line
    midpoint_x = 1/2 * (lda.means_[0][0] + lda.means_[1][0])
    midpoint_y = 1/2 * (lda.means_[0][1] + lda.means_[1][1])
    midpoint = [midpoint_x, midpoint_y]
    
    #standard line
    def line(x, get_eq=False):
        m = (line_point1[1] - line_point2[1])/(line_point1[0] - line_point2[0])
        b = line_point1[1] - m*line_point1[0]
        if get_eq:
            return m, b
        else:
            return m*x + b
    
    #extract lines parallel to the separating line    
    def parallel_line(x, ref, get_eq = False):
        m = (line_point1[1] - line_point2[1])/(line_point1[0] - line_point2[0])
        b = ref[1] - m*ref[0]
        if get_eq:
            return m, b
        else:
            return m*x + b
    
    #extract line perpendicular to the separating line, located equidistantly between the class means
    def perpendicular_line(x, get_eq=False):
        m, b = line(0, True)
        m2 = -1/m
        b2 = midpoint[1] - m2*midpoint[0]
        if get_eq:
            return m2, b2
        else:
            return m2*x + b2

    #get projection of data points onto perpendicular line
    def get_projection(ref):
        m, b = perpendicular_line(0, True)
        m2, b2 = parallel_line(0, ref, True)
        x = (b2 - b) / (m - m2)
        y = parallel_line(x, ref)
        return [x, y]


    z = perpendicular_line(np.linspace(-1, 1))
    
    ax.plot(np.linspace(-1,1), z, '--', color = "k", linewidth = 1)

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((-1.1, 2.1))
    
    #draw projection lines
    intersect_new = get_projection(mean_point2)
    intersect_new2 = get_projection(mean_point1)
    ax.plot(intersect_new[0], intersect_new[1], 'go', label='projection', markersize = 5)
    ax.plot(intersect_new2[0], intersect_new2[1], 'go', label='projection', markersize = 5)
    ax.plot([intersect_new[0], mean_point2[0]], [intersect_new[1], mean_point2[1]], color='green', label='distance', linewidth = 1) 
    ax.plot([intersect_new2[0], mean_point1[0]], [intersect_new2[1], mean_point1[1]], color='green', label='distance', linewidth = 1) 
    
    example_data_point = X0_fp[1]
    intersect_data = get_projection(example_data_point)
    ax.plot(intersect_data[0], intersect_data[1], 'o', color ='brown', markersize = 5)
    ax.plot([intersect_data[0], example_data_point[0]], [intersect_data[1], example_data_point[1]], color='brown', label='distance', linewidth = 1)
    
    return fig, splots


for i, (X, y) in enumerate([dataset_fixed_cov()]):
    # fit and plot lda
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    fig, splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)

    
    # fit svm 
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    
    ax = splot[1]
    ax.set_title("Support Vector Machine (Linear Kernel)", fontsize = 11)
    ax.scatter(X[:, 0], X[:, 1], c= ["red" if y[i] == 0 else "steelblue" for i in range(len(y))], s = 10)#c=y, s=30, cmap='Paired')
    

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha = 0.7, linewidths = 1,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=40,
               linewidth=1, facecolors='none', edgecolors='k', alpha = 0.8)
    
    ax.set_ylim((-1.1, 2.1))
        
    ax.tick_params(
    axis='y',          
    which='both',      
    bottom=False,      
    top=False,         
    labelleft=False) 
    
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

f = plt.gcf()
f.set_size_inches(6.2,2.5)