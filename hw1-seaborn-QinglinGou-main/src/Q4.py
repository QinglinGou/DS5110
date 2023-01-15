import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Q3. Get tidy dataset
s = pd.read_csv('https://raw.githubusercontent.com/ds5110/rdata/main/data/Smarket.csv')
new_smarket = s.melt(id_vars =["Year","Volume","Direction"],
            var_name = "Lag", 
            value_name = "Percentage change in S&P")
new_smarket.columns =['Year','Volume',"Today's Direction","Day","Percentage change in S&P"]
print(new_smarket.head())

# Q4.Recreate Figure 1.2 with Seaborn using the tidy form of the Smarket dataset.

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
plt.show()
