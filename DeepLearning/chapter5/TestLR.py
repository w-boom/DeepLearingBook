""" 测试线性回归 """
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats # linear Regression
from LinearRegression import MyLinRegress


def GetValues(x, slope, intercept):
    x = np.asarray(x)
    return slope * x + intercept
  
def GetFigure(x, y, values):
    plt.scatter(x, y)
    plt.plot(x, values)
    plt.show()
    pass
  
def main():  
    x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
    y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
    
    """
    slope : float
        Slope of the regression line.
    intercept : float
        Intercept of the regression line.
    rvalue : float
        Correlation coefficient.
    pvalue : float
        The p-value for a hypothesis test whose null hypothesis is
        that the slope is zero, using Wald Test with t-distribution of
        the test statistic. See `alternative` above for alternative
        hypotheses.
    stderr : float
        Standard error of the estimated slope (gradient), under the
        assumption of residual normality.
    intercept_stderr : float
        Standard error of the estimated intercept, under the assumption
        of residual normality.
    """
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    
    slope1, intercept1 = MyLinRegress(x, y)

    values = GetValues(x, slope, intercept)
    GetFigure(x, y, values)

    values1 = GetValues(x, slope1, intercept1)
    GetFigure(x, y, values1)

    pass
    

if __name__ == '__main__':
    main()