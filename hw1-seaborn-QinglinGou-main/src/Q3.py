import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Q3.Put a dataframe with the Smarket dataset into tidy form. Use .head() to print the first few lines the result.
s = pd.read_csv('https://raw.githubusercontent.com/ds5110/rdata/main/data/Smarket.csv')
new_smarket = s.melt(id_vars =["Year","Volume","Direction"],
            var_name = "Lag", 
            value_name = "Percentage change in S&P")
new_smarket.columns =['Year','Volume',"Today's Direction","Day","Percentage change in S&P"]

print(new_smarket.head())
