import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
df = pd.read_csv("https://raw.githubusercontent.com/ds5110/rdata/main/data/flights.csv")
flight=df[df['distance']<3000]
flight= flight.dropna()

flightdest = flight.groupby(by='dest').mean()
sns.regplot(data=flightdest, x='distance', y='arr_delay')
plt.savefig('figure3')
plt.show()



