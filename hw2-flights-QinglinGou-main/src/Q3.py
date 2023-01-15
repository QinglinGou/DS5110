import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

df = pd.read_csv("https://raw.githubusercontent.com/ds5110/rdata/main/data/flights.csv")
flight=df[df['distance']<3000]
flight= flight.dropna()

count = flight.groupby(['year', 'month','day','hour', 'flight']).size().reset_index(name = 'n').n.value_counts()
print(count)
count2= flight.groupby(['year', 'month','day','hour', 'flight', 'carrier']).size().reset_index(name = 'n').n.value_counts()
print(count2)