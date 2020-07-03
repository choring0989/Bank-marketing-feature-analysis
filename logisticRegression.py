# Importing data analysis libraries
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

x = x.drop(columns = ['age', 'job', 'marital', 'month'], axis=1)

#sns.boxplot(data=x)
#plt.show()

# Divide data set
from sklearn.model_selection import train_test_split
dfx_train, dfx_test, dfy_train, dfy_test = train_test_split(x, y, test_size=0.20, random_state=1)
dfx_train, dfx_ev, dfy_train, dfy_ev = train_test_split(dfx_train, dfy_train, test_size=0.20, random_state=1)

print("Size of training set: {} / Size of validation set: {} / Size of test set:"
      " {}\n".format(dfx_train.shape[0], dfx_ev.shape[0], dfx_test.shape[0]))

# Create a parameter grid
param_grid = {'C': [0.008, 0.01, 0.03, 0.05, 0.07],
              'solver': ["liblinear", "newton-cg", "lbfgs", "sag", "saga"]}

# Create cross-validation grid objects
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, return_train_score=True)

# Automatically create new models for the entire training data set with optimal parameters
grid_search.fit(dfx_train, dfy_train)
print("Test score: {:.4f}".format(grid_search.score(dfx_test, dfy_test)))
print("Best params:", grid_search.best_params_)
print("Highest cross-validation score: {:.4f}".format(grid_search.best_score_))
print("Best estimator:\n", grid_search.best_estimator_)

results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results.mean_test_score).reshape(5, 5)

# Visualization
cmap = sns.cubehelix_palette(8)
sns.heatmap(scores, xticklabels=param_grid['solver'], yticklabels=param_grid['C'], cmap=cmap, annot=True)
plt.show()

# Make Logistic Regression model
LR_model = LogisticRegression(C=0.105, solver='sag', max_iter=100).fit(dfx_train, dfy_train.values.ravel())

#Accuracy calculation
from sklearn.metrics import accuracy_score
y_pred_LR = LR_model.predict(dfx_test)
print("Logistic Regression model's best accuracy", accuracy_score(dfy_test, y_pred_LR))

matrix1 = pd.crosstab(dfy_test.values.ravel(), y_pred_LR, rownames=['Actual'], colnames=['Predicted'], margins=True)
sns.heatmap(matrix1, annot=True, fmt="d")
plt.title("Heatmap")
plt.show()
