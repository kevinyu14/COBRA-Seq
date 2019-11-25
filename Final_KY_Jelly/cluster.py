import scipy
import numpy as np

data = np.loadtxt('results.txt.gz')
dimf = open('dimensions_of_results.txt.gz', 'r')
dimnames = dimf.readlines()
