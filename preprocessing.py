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

# Self label encoding -> This is a manual label encoding to detect unknown data or continuous data.
def LabelEncoding(data):
    data.job.replace(('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'),(1,2,3,4,5,6,7,8,9,10,11,-1),inplace=True)
    data.default.replace(('yes','no', 'unknown'),(1,0,-1), inplace=True)
    data.housing.replace(('yes','no', 'unknown'),(1,0,-1), inplace=True)
    data.loan.replace(('yes','no', 'unknown'),(1,0,-1), inplace=True)
    data.marital.replace(('married','single','divorced', 'unknown'),(2,1,3,-1), inplace=True)
    data.contact.replace(('telephone','cellular','unknown'),(1,2,-1), inplace=True)
    data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'), (1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)
    data.day_of_week.replace(('mon','tue','wed','thu','fri'), (1,2,3,4,5),inplace=True)
    data.poutcome.replace(('failure','nonexistent','success'),(0,-1,1),inplace=True)
    # Continuous data
    data.education.replace(('basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'), (2,3,4,5,1,7,6,-1),inplace=True)
    # Convert 999 to -1
    data.pdays.replace((999), (-1), inplace=True)
    return data

bank = pd.read_csv('bank-additional-full.csv', sep = ';')
#Converting dependent variable categorical to dummy
x = bank.drop(columns = ['duration','y'], axis=1)
y = pd.DataFrame(bank.y, columns=['y'])

y.y.replace(('yes','no'), (1,0), inplace=True)

# Do label encoding
x = LabelEncoding(x)
x = x.rename(columns={'emp.var.rate': 'emp_var_rate', 'cons.price.idx': 'cons_price_idx', 'cons.conf.idx': 'cons_conf_idx', 'nr.employed': 'nr_employed'})
label = x.columns
print(label)

# Export preprocessing data for use in wise prophet
bank.to_csv('bank.csv', sep=',', na_rep='NaN', index=False)

# print(bank.describe().T)
# print(bank)
# print(y.describe().T)

# Measure missing data
import missingno as msno
msno.matrix(bank)
plt.show()

cmap = sns.cubehelix_palette(8)
sns.heatmap(x.corr(), annot=True, cmap=cmap)
plt.show()

# Check data distribution
sns.boxplot(data=x)
plt.show()

# Eliminate Outliers Using OLS Models
import statsmodels.api as sm
labels = ["age", "campaign", "pdays", "previous", 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx','euribor3m', 'nr_employed']
for i in labels:
    model = sm.OLS(y, x[i].astype(float))
    result = model.fit()
    print(len(result.resid_pearson), len(x))
    k=len(result.resid_pearson)-1
    for j in reversed(result.resid_pearson):
        if(j > 3.1 or j < -1):
            if(y.y.iloc[k] != 1):
                x = x.drop(x.index[k])
                y = y.drop(y.index[k])
        k=k-1
    print(result.resid_pearson)
    plt.figure(figsize=(10, 4))
    plt.stem(result.resid_pearson[34980:35280])
    plt.axhline(3, c="g", ls="--")
    plt.axhline(-1, c="g", ls="--")
    plt.title("Standardized residuals wtih "+i)
    plt.show()

print(x.describe().T)
# Export
x.to_csv('preprocessing_x.csv', sep=',', na_rep='NaN', index=False)
y.to_csv('preprocessing_y.csv', sep=',', na_rep='NaN', index=False)
