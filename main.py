import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print('\n1--------------------------')
data_frame = pd.read_csv('D:/DZ/11sem/AI_Enregy/LR3/income.csv',  encoding = "ISO-8859-1")
print(data_frame)




print('\n2.1--------------------------')

# Два способа посчитать null значения
# 1) путем замены принятого обозначения " ?" на объект класса NoneType и подсчет с помощью метода pd
# test = data_frame.replace([" ?"], None, inplace = True)
# test.to_csv('D:/DZ/11sem/AI_Enregy/LR3/test.csv')
#
# print(test["workclass"].value_counts())
#
# print(test.isnull().sum())

# 2) путем подсчета принятого обозначения " ?"
# print((data_frame == " ?").sum())
null_vals = ((data_frame == " ?").sum().loc[lambda x: x > 0])
print('Gризнаки, в которых есть пропущенные значенмя {}\n'.format(null_vals.index.values))

print(null_vals)
no_null_df = data_frame.replace([" ?"], "No info")

print('\n2.2--------------------------')

target = data_frame["workclass"].replace([" ?"], "No info").hist(figsize=(10, 8), bins=range(1,10), edgecolor = 'black', align='left', rwidth=1, grid = True)
target.plot()
plt.title('Workclass Values')
plt.ylabel('Frequency')
plt.show()


print('\n2.3--------------------------')

f, ax = plt.subplots(figsize=(10, 8))
ax = sns.countplot(x="income", hue="sex", data=no_null_df, palette="Set1")
ax.set_title("Frequency distribution of income variable wrt sex")
plt.grid(True)
plt.ylabel('Frequency')

plt.show()

print('\n2.4--------------------------')

f, ax = plt.subplots(figsize=(10, 8))
ax = sns.countplot(x="income", hue="race", data=no_null_df, palette="Set1")
ax.set_title("Frequency distribution of income variable wrt race")
plt.grid(True)
plt.ylabel('Frequency')

plt.show()

print('\n2.5--------------------------')

f, ax = plt.subplots(figsize=(10, 8))
ax = sns.countplot(x="workclass", hue="income", data=no_null_df, palette="Set1")
ax.set_title("Frequency distribution of Workclass variable wrt income")
plt.grid(True)
plt.ylabel('Frequency')

plt.show()

print('\n2.6--------------------------')

f, ax = plt.subplots(figsize=(10, 8))
ax = sns.countplot(x="workclass", hue="sex", data=no_null_df, palette="Set1")
ax.set_title("Frequency distribution of Workclass variable wrt income")
plt.grid(True)
plt.ylabel('Frequency')

plt.show()

print('\n2.7--------------------------')

# f, ax = plt.subplots(figsize=(10,8))
# x = no_null_df['age']
# ax = sns.distplot(x, bins=10, color='blue')
# ax.set_title("Distribution of age variable")
# plt.show()

target_a = data_frame['age'].hist(figsize=(10, 8), bins=range(17,91), edgecolor = 'black', align='left', rwidth=1, grid=True)
target_a.plot()
plt.title('Age Values')
plt.ylabel('Frequency')
plt.show()

print('\n2.8--------------------------')

f, ax = plt.subplots(figsize=(10,8))
x = no_null_df['age']
ax = sns.boxplot(x)
ax.set_title("Visualize outliers in age variable")
plt.show()

print('\n2.9--------------------------')

f, ax = plt.subplots(figsize=(10, 8))
ax = sns.boxplot(x="income", y="age", data=no_null_df)
ax.set_title("Visualize income wrt age variable")
plt.show()

print('\n2.10--------------------------')

f, ax = plt.subplots(figsize=(10, 8))
ax = sns.boxplot(x="income", y="age", hue="sex", data=no_null_df)
ax.set_title("Visualize income wrt age and sex variable")
ax.legend(loc='upper right')
plt.show()

print('\n2.11--------------------------')

f, ax = plt.subplots(figsize=(10, 8))
ax = sns.boxplot(x="race", y="age", data=no_null_df)
ax.set_title("Visualize race wrt age variable")
plt.show()


print('\n2.12--------------------------')

# data_frame.select_dtypes('number').corr().style.format("{:.4}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)

corr = data_frame.select_dtypes('number').corr()
print(corr)
dataplot = sns.heatmap(corr)
plt.show()

print('\n2.13--------------------------')
print(data_frame)
data_frame.replace(' ?', np.NaN, inplace=True)
data_frame.to_csv('D:/DZ/11sem/AI_Enregy/LR3/test.csv')

print('\n2.14--------------------------')

categorical = [var for var in data_frame.columns if data_frame[var].dtype == 'O']
print(data_frame[categorical].keys())
print(data_frame[categorical].head())

print('\n2.15--------------------------')

number = [var for var in data_frame.columns if data_frame[var].dtype != 'O']
print(data_frame[number].keys())
print(data_frame[number].head())

print('\n3--------------------------')