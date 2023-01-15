import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

# Q1.Use Seaborn to recreate Figure 1.1 in the book, including the models, with all 3 subplots in the same figure.

df = pd.read_csv('https://raw.githubusercontent.com/ds5110/rdata/main/data/Wage.csv')
df['education']= df['education'].apply(lambda x:x[0])

fig, ax = plt.subplots(1,3,figsize=(15,7))
a1 = sns.regplot(x=df['age'],y=df['wage'],lowess=True,scatter_kws={'color':'grey'},line_kws={'color':'blue'},ax=ax[0])
a1.set_xticks(ticks=[20,40,60,80])
a2 = sns.regplot(x=df['year'],y=df['wage'],lowess=True,scatter_kws={'color':'grey'},line_kws={'color':'blue'},ax=ax[1])
a2.set_xticks(ticks=[2003,2006,2009])
my_pal = {'1':'skyblue','2':'g','3':'y','4':'b','5':'orange'}
sns.boxplot(x=df['education'],y=df['wage'],order=['1','2','3','4','5'],palette=my_pal,medianprops=dict(color="black",alpha=1,linewidth=3.0,), whiskerprops = dict(linestyle='--',color='black'))
plt.show()
