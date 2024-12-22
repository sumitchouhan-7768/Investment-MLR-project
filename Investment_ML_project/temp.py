import numpy as np 
import pandas as import pd 
import matplotlib.pyplot as plt 

df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\NIT All-Projects\Investment_ML_project\Investment.csv")

x = df.iloc[:,:-1]
y = df.iloc[:,4]


