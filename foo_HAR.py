import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

url = "train.csv"
df = pd.read_csv(url,   nrows=1)
columns = list(df.head(0))
print(columns)

df = pd.read_csv(url, skiprows=1, names=columns)
#print("len(columns)",len(columns))

#print(df.head(5))
features = ["fBodyAccJerk.sma","tBodyAccJerkMag.sma", "fBodyBodyAccJerkMag.mean", "tBodyGyroJerk.sma", "tBodyGyroJerkMag.sma", 
                "tBodyAccJerk.std.X", "tBodyAcc.sma" ]
print("len(features)",len(features))
X = df.loc[:, features].values

# Standardizing the features
X = StandardScaler().fit_transform(X)
# Separating out the target
#y = df.loc[:,['target']].values

K = range(1,10)

  # scipy.cluster.vq.kmeans
KM = [kmeans(X,k) for k in K] # apply kmeans 1 to 10
centroids = [cent for (cent,var) in KM]   # cluster centroids

D_k = [cdist(X, cent, 'euclidean') for cent in centroids]

cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]

print("cIdx", cIdx)
print("dist", dist)

avgWithinSS = [sum(d)/X.shape[0] for d in dist]


kIdx = 2
# plot elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
      markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
tt = plt.title('Elbow for K-Means clustering')
plt.show()


import matplotlib.pyplot as plt1
from sklearn.cluster import KMeans
km = KMeans(3, init='k-means++') # initialize
km.fit(X)
c = km.predict(X) # classify into three clusters
print("ccccc", len(c))
# see the code in helper library kmeans.py
# it wraps a number of variables and maps integers to categoriy labels
# this wrapper makes it easy to interact with this code and try other variables
# as we see below in the next plot
import kmeans as mykm
(pl0,pl1,pl2) = mykm.plot_clusters(X,c,3,2) # column 3 GDP, vs column 2 infant mortality. Note indexing is 0 based

'''
fig, axes = plt1.subplots(nrows=10, ncols=2, figsize=(20, 13))
ax=axes.ravel()# flat axes with numpy ravel

for i in range(1,len(features)):
    item1 = [ d[i] for d in X if d[i] != 0]
    item2 = [ d[0] for d in X if d[0] != 0]
    item = item1 + item2
    low = min(item)

    high = max(item)
    #print(low)
    #print(high)
    bins = np.linspace(low, high, 20)
    ax[i].hist(item1, bins, alpha=0.6,  label=features[i])
    ax[i].hist(item2, bins, alpha=0.6,  label=features[0])
    #ax[i].hist(item1, bins,  color='r', label="one")
    #ax[i].hist(item2, bins,  color='g', label="zero")
    ax[i].legend(loc='upper right')
    #ax[i].axes.get_xaxis().set_visible(False)
    #ax[i].set_yticks(())
plt1.tight_layout()
plt1.show()

 tBodyAccJerkMag.iqr      tBodyAccJerkMag.max             tBodyAcc.sma 
               0.3296044                0.3300089                0.3306103 
     fBodyAccJerk.mean.X       tBodyAccJerk.mad.X       tBodyAccJerk.std.X 
               0.3308718                0.3313329                0.3314136 
 fBodyBodyAccJerkMag.mad    tBodyGyroJerkMag.mean     tBodyGyroJerkMag.sma 
               0.3317676                0.3326879                0.3326879 
        fBodyAccMag.mean          fBodyAccMag.sma        tBodyGyroJerk.sma 
               0.3326945                0.3326945                0.3346974 
     tBodyAccJerkMag.std      tBodyAccJerkMag.mad fBodyBodyAccJerkMag.mean 
               0.3355263                0.3355557                0.3373821 
 fBodyBodyAccJerkMag.sma     tBodyAccJerkMag.mean      tBodyAccJerkMag.sma 
               0.3373821                0.3414347                0.3414347 
        tBodyAccJerk.sma            fBodyGyro.sma         fBodyAccJerk.sma 
               0.3420842                0.3425234                0.3427119 
            fBodyAcc.sma 
               0.3427775 
'''
