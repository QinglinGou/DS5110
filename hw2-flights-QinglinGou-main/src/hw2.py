
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
df = pd.read_csv("https://raw.githubusercontent.com/ds5110/rdata/main/data/flights.csv")
flight=df[df['distance']<3000]
flight= flight.dropna()
#Q1
sns.jointplot(data=flight, x="distance", y="arr_delay")
plt.savefig('figure1')
plt.show()
sns.jointplot(data=flight, x="distance", y="arr_delay",hue="carrier")
plt.savefig('figure2')
plt.show()
#Q2
flightdest = flight.groupby(by='dest').mean()
sns.regplot(data=flightdest, x='distance', y='arr_delay')
plt.savefig('figure3')
plt.show()
#Q3
count = flight.groupby(['year', 'month','day','hour', 'flight']).size().reset_index(name = 'n').n.value_counts()
print(count)
count2= flight.groupby(['year', 'month','day','hour', 'flight', 'carrier']).size().reset_index(name = 'n').n.value_counts()
print(count2)
#Q4
con = sqlite3.connect('mydb.sqlite')
cur = con.cursor()
cur.execute("SELECT f.carrier,f.tailnum, f.month, f.day FROM planes AS p INNER JOIN flights AS f ON p.tailnum=f.tailnum WHERE p.manufacturer ='AIRBUS INDUSTRIE' ORDER BY  f.carrier,f.tailnum,f.month,f.day  LIMIT 10")
for row in cur:
  print(row)



