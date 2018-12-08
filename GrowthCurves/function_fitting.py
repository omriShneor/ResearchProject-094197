import math
import numpy as np
import matplotlib.pyplot as plt

def polyniomial_fitting(expected_weight_df,x_index,y_index,polynomial_rank):
    x = expected_weight_df[x_index].values
    y = expected_weight_df[y_index].values
    p = np.polyfit(x,y,polynomial_rank)
    y1 = np.polyval(p,x)
    return (x,y1)

