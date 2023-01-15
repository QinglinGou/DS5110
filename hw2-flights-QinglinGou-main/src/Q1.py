import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

df = pd.read_csv("https://raw.githubusercontent.com/ds5110/rdata/main/data/flights.csv")
flight=df[df['distance']<3000]
flight= flight.dropna()

sns.jointplot(data=flight, x="distance", y="arr_delay")
plt.savefig('figure1')
plt.show()
sns.jointplot(data=flight, x="distance", y="arr_delay",hue="carrier")
plt.savefig('figure2')
plt.show()