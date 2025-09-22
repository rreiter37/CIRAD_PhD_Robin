from Scripts_python.utils.utils_bdd import split_data
import pandas as pd
import numpy as np

mode = "Regression"
data_source = "grapevine_chloride_260_KS"
Xcal, Ycal, Xval, Yval = split_data(mode, data_source, verbose=True)

print(Xval.shape)
print(Xval)
print(Yval.shape)



mode = "Regression"
data_source = "BeerOriginalExtract"
Xcal, Ycal, Xval, Yval = split_data(mode, data_source, verbose=True)

print(Xval.shape)
print(Xval)
print(Yval.shape)