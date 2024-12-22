# K- mean clustring
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\HP\Downloads\Mall_Customers.csv")
X = df.iloc[:,[3,4]].values

#using elbow method
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    Kmeans = KMeans(n_clusters =i,init="k-means++",random_state = 0)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()

Kmeans = KMeans(n_clusters = 5, init = 'k-means++',random_state = 0)
y_Kmeans = Kmeans.fit_predict(X)


