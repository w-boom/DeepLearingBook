import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the inbuilt dataset of scikit learn, breast cancer wisconsin dataset.
from sklearn.datasets import load_breast_cancer

def main():
    data = load_breast_cancer()
    
if __name__ == "__main__":
    main()