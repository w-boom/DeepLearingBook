import numpy as np

"""
    Linear Regression
syntax:
    slope, intercept = MyLinRegress(x, y)
"""

def MyLinRegress(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    xMean = np.mean(x, None)
    yMean = np.mean(y, None)
    
    # Average sums of square differences from the mean
    # ssX = mean( (x-mean(x))^2 )
    # ssXY = mean( (x-mean(x)) * (y-mean(y)) )
    ssX, ssY, _, _ = np.cov(x, y, bias=1).flat
    
    slope = ssX / ssY
    intercept = yMean - slope * xMean
    
    return slope, intercept
    