# analyzing_movements
Repository supplementing the paper "Supervised learning for analyzing movement patterns in  a virtual reality experiment" by Vogel et al.

he folder traindata contains preprocessed data of the experiment in pythonic "pickle" format. The original data can be found at https://osf.io/rnz62. Preprocessed in this case means:
For the base study, data has been cut to have an equal number of time points per participant. A few participants had to be removed due to missing or defective tracking. 
For the evaluation study, a few participants had to be removed due to missing data. 

The module eval_base_study contains all methods for reproducing the results listed in the Method and Results section of said paper. 
Calling

python eval_base_study.py

will produce two sets dictionaries, "whole_session_results.npy" and "segment_results.npy", accessable in the traindata-subfolder.
 
However, it is not recommend to call this module from terminal, as it TAKES LONG. Please choose a Python IDE and run the scripts and/or single methods there.

The dictionaries contain the results
of 100 runs of testing and training as described in the paper. In short:
In each run, a maximum number of 12 correct predictions can be obtained (per body part).
The dictionaries represent check lists about how often a certain number of correct predictions (i.e., a number <= 12) appeared throughout the 100 runs.
This is checked for left hand, right hand and both hand patterns.
An entry in the dictionary is of the form {'n': [a,b,c]}, where n represents a certain number of correct predictions (e.g., '10') and a is the number of appearences of n 
throughout the 100 runs for left hand, b the number for right hand, and c the number for both hand patterns. 

Please see the docs of methods "prepare_results_for_plot_whole_sessions" and "prepare_results_for_plot_segments" to see, which dictionary represents the results of which
algorithm. 

Note: Since selection of training and testing sets as well as choice of segments is random, your results will most likely not match the ones listed in the paper exactly. 

The module eval_validation_study.py analogously contains methods for reproducing the results listed int the Reproducibility section of the paper. 
