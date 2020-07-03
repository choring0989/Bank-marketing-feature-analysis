import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth', -1)
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 1000)

x = pd.read_csv('preprocessing_x.csv')
y = pd.read_csv('preprocessing_y.csv')

label = x.columns

# Scale the data
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns = label)
#print(x.describe().T)

# Divide data set
from sklearn.model_selection import train_test_split
dfx_train, dfx_test, dfy_train, dfy_test = train_test_split(x, y, test_size=0.20, random_state=1)
dfx_train, dfx_ev, dfy_train, dfy_ev = train_test_split(dfx_train, dfy_train, test_size=0.20, random_state=1)

print("Size of training set: {} / Size of validation set: {} / Size of test set:"
      " {}\n".format(dfx_train.shape[0], dfx_ev.shape[0], dfx_test.shape[0]))

# Lower data dimensions with PCA
from sklearn.decomposition import PCA
import scipy as sp
pca= PCA(n_components=1)
X_principal = pca.fit_transform(dfx_train)
X_principal=X_principal.reshape(-1,1)

X_principalq = pca.fit_transform(dfx_test)
X_principalq=X_principalq.reshape(-1,1)

from sklearn.neighbors import KNeighborsClassifier
import mglearn

# Calculate accuracy by generating KNN model based on K number
dfy_test = dfy_test.reset_index(drop=True)
dfy_test = dfy_test.values
means = []
q=[1, 3, 5, 9, 11, 13, 15, 17, 19]
for n_neighbors in q:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_principal, dfy_train.values.ravel())
    prediction = clf.predict(X_principalq)
    l=0
    curmean = 0
    for n in prediction:
        if(n == dfy_test[l]):
            curmean = curmean+1
        l=l+1
    print("K:", n_neighbors, "   Acurracy:", curmean/len(prediction))
    means.append(curmean/len(prediction))

print(means)

# Visualization
plt.plot(q, means, marker="o")
plt.title("knn")
plt.show()