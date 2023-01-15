import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.show()
