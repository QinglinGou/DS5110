import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

df = pd.read_csv("https://raw.githubusercontent.com/ds5110/rdata/main/data/flights.csv")
flight=df[df['distance']<3000]
flight= flight.dropna()

con = sqlite3.connect('mydb.sqlite')
cur = con.cursor()
cur.execute("SELECT f.carrier,f.tailnum, f.month, f.day FROM planes AS p INNER JOIN flights AS f ON p.tailnum=f.tailnum WHERE p.manufacturer ='AIRBUS INDUSTRIE' ORDER BY  f.carrier,f.tailnum,f.month,f.day  LIMIT 10")
for row in cur:
  print(row)