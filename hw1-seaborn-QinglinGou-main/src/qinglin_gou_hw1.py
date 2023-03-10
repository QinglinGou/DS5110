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

# Q2.Use Seaborn with the Smarket dataset (in the rdata repo) to recreate Figure 1.2 in the book from the original (messy) dataframe.
smarket = pd.read_csv('https://raw.githubusercontent.com/ds5110/rdata/main/data/Smarket.csv')
smarket.columns = ["Year","Yesterday","Two Days Previous","Three Days Previous","Four Days Previous","Five Days Previous","Volume", "Today","Today's Direction"]
fig, ax = plt.subplots(1,3,figsize=(15,7))
pat={"Down":"skyblue","Up":"orange"}
a4=sns.boxplot(x=smarket["Today's Direction"],y=smarket['Yesterday'],order=['Down','Up'],palette=pat,ax=ax[0],medianprops=dict(color="black",alpha=1,linewidth=4.0,), whiskerprops = dict(linestyle='--',color='black'))
a4.set_title('Yesterday')
a4.set_ylabel('Percentage change in S&P')
a5=sns.boxplot(x=smarket["Today's Direction"],y=smarket['Two Days Previous'],order=['Down','Up'],palette=pat,ax=ax[1],medianprops=dict(color="black",alpha=1,linewidth=4.0,), whiskerprops = dict(linestyle='--',color='black'))
a5.set_title('Two Days Previous')
a5.set_ylabel('Percentage change in S&P')
a6=sns.boxplot(x=smarket["Today's Direction"],y=smarket['Three Days Previous'],order=['Down','Up'],palette=pat,ax=ax[2],medianprops=dict(color="black",alpha=1,linewidth=4.0,), whiskerprops = dict(linestyle='--',color='black'))
a6.set_title('Three Days Previous')
a6.set_ylabel('Percentage change in S&P')

# Q3.Put a dataframe with the Smarket dataset into tidy form. Use .head() to print the first few lines the result.
s = pd.read_csv('https://raw.githubusercontent.com/ds5110/rdata/main/data/Smarket.csv')
new_smarket = s.melt(id_vars =["Year","Volume","Direction"],
            var_name = "Lag", 
            value_name = "Percentage change in S&P")
new_smarket.columns =['Year','Volume',"Today's Direction","Day","Percentage change in S&P"]
new_smarket.head()

# Q4.Recreate Figure 1.2 with Seaborn using the tidy form of the Smarket dataset."""

a=new_smarket.loc[(new_smarket.Day== "Lag1") ]
b=new_smarket.loc[(new_smarket.Day== "Lag2") ]
c=new_smarket.loc[(new_smarket.Day== "Lag3") ]

fig, ax = plt.subplots(1,3,figsize=(15,7))
pat={"Down":"skyblue","Up":"orange"}
b1=sns.boxplot(x=a["Today's Direction"],y=a['Percentage change in S&P'],order=['Down','Up'],palette=pat,ax=ax[0],medianprops=dict(color="black",alpha=1,linewidth=4.0,), whiskerprops = dict(linestyle='--',color='black'))
b1.set_title('Yesterday')
b1.set_ylabel('Percentage change in S&P')
b2=sns.boxplot(x=b["Today's Direction"],y=b['Percentage change in S&P'],order=['Down','Up'],ax=ax[1],palette=pat,medianprops=dict(color="black",alpha=1,linewidth=4.0,), whiskerprops = dict(linestyle='--',color='black'))
b2.set_title('Two Days Previous')
b2.set_ylabel('Percentage change in S&P')
b3=sns.boxplot(x=c["Today's Direction"],y=c['Percentage change in S&P'],order=['Down','Up'],ax=ax[2],palette=pat,medianprops=dict(color="black",alpha=1,linewidth=4.0,), whiskerprops = dict(linestyle='--',color='black'))
b3.set_title('Three Days Previous')
b3.set_ylabel('Percentage change in S&P')


print(new_smarket.head())
plt.show()
