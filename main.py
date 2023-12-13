import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


def find_optimal_param(param_range, X_train, X_test, Y_train, Y_test, model_name):
    f_score_test = []
    f_score_train = []
    param = ''
    for k in param_range:
        if model_name == 'dtc':
            model = DecisionTreeClassifier(max_depth=k)  # создание модели с заданной метрикой
            param = 'depth'
        if model_name == 'rfc':
            model = RandomForestClassifier(n_estimators=k)
            param = 'n_estim.'
        if model_name =='cbc':
            model = CatBoostClassifier(n_estimators=k)
            param = 'n_estim.'

        model.fit(X_train, Y_train)  # Обучение модели
        # print("our accuracy is:{}".format(multNB.score(X_train, Y_train)))

        Y_predict = model.predict(X_test)
        Y_predict_train = model.predict(X_train)

        f_score_test.append(f1_score(Y_test, Y_predict))
        f_score_train.append(f1_score(Y_test, Y_predict))

    matrix = np.matrix(np.c_[param_range, f_score_test, f_score_train])
    models = pd.DataFrame(data=matrix,
                          columns=[param, 'test F-Measure', 'train F-Measure'])

    best_index = models['test F-Measure'].idxmax()
    best_value = models.iloc[best_index]['test F-Measure']

    print('Best F-measure for {} using depth value {} index {} '.format(model_name, best_value, best_index))

    plt.subplot(1, 2, 1)
    plt.plot(param_range, f_score_test)
    plt.plot(param_range[best_index], best_value, marker='o', color='red')
    plt.annotate(xy=(param_range[best_index], best_value),
                 text='({}, {})'.format(("%.1f" % best_value), ("%.1f" % param_range[best_index])))
    plt.title("Test F_Measure", fontsize=10)
    plt.ylabel('Accuracy score(%)', fontsize=8)
    plt.xlabel(param, fontsize=8)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(param_range, f_score_train)
    plt.plot(param_range[best_index], best_value, marker='o', color='red')
    plt.annotate(xy=(param_range[best_index], best_value),
                 text='({}, {})'.format(("%.1f" % best_value), ("%.1f" % param_range[best_index])))
    plt.title("Train F_Measure", fontsize=10)
    plt.ylabel('Accuracy score(%)', fontsize=8)
    plt.xlabel(param, fontsize=8)
    plt.grid(True)

    plt.show()

    if model_name == 'dtc':
        optimal_model = DecisionTreeClassifier(max_depth=param_range[best_index])  # создание модели с заданной метрикой
    if model_name == 'rfc':
        optimal_model = RandomForestClassifier(n_estimators=param_range[best_index])
    if model_name == 'cbc':
        optimal_model = CatBoostClassifier(n_estimators=param_range[best_index])

    optimal_model.fit(X_train, Y_train)

    confusion_matrix_opt = confusion_matrix(Y_test, optimal_model.predict(X_test))
    conf_matrix = pd.DataFrame(data=confusion_matrix_opt, columns=['predicted ham', 'predicted spam'],
                               index=['actual ham', 'actual spam'])

    print(conf_matrix)


print('\n1--------------------------')
data_frame = pd.read_csv('D:/DZ/11sem/AI_Enregy/LR3/income.csv',  encoding = "ISO-8859-1")
print(data_frame)
data_frame.dropna(inplace=True,axis=0)




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

# print('\n2.2--------------------------')
#
# target = data_frame["workclass"].replace([" ?"], "No info").hist(figsize=(10, 8), bins=range(1,10), edgecolor = 'black', align='left', rwidth=1, grid = True)
# target.plot()
# plt.title('Workclass Values')
# plt.ylabel('Frequency')
# plt.show()
#
#
# print('\n2.3--------------------------')
#
# f, ax = plt.subplots(figsize=(10, 8))
# ax = sns.countplot(x="income", hue="sex", data=no_null_df, palette="Set1")
# ax.set_title("Frequency distribution of income variable wrt sex")
# plt.grid(True)
# plt.ylabel('Frequency')
#
# plt.show()
#
# print('\n2.4--------------------------')
#
# f, ax = plt.subplots(figsize=(10, 8))
# ax = sns.countplot(x="income", hue="race", data=no_null_df, palette="Set1")
# ax.set_title("Frequency distribution of income variable wrt race")
# plt.grid(True)
# plt.ylabel('Frequency')
#
# plt.show()
#
# print('\n2.5--------------------------')
#
# f, ax = plt.subplots(figsize=(10, 8))
# ax = sns.countplot(x="workclass", hue="income", data=no_null_df, palette="Set1")
# ax.set_title("Frequency distribution of Workclass variable wrt income")
# plt.grid(True)
# plt.ylabel('Frequency')
#
# plt.show()
#
# print('\n2.6--------------------------')
#
# f, ax = plt.subplots(figsize=(10, 8))
# ax = sns.countplot(x="workclass", hue="sex", data=no_null_df, palette="Set1")
# ax.set_title("Frequency distribution of Workclass variable wrt income")
# plt.grid(True)
# plt.ylabel('Frequency')
#
# plt.show()
#
# print('\n2.7--------------------------')
#
# # f, ax = plt.subplots(figsize=(10,8))
# # x = no_null_df['age']
# # ax = sns.distplot(x, bins=10, color='blue')
# # ax.set_title("Distribution of age variable")
# # plt.show()
#
# target_a = data_frame['age'].hist(figsize=(10, 8), bins=range(17,91), edgecolor = 'black', align='left', rwidth=1, grid=True)
# target_a.plot()
# plt.title('Age Values')
# plt.ylabel('Frequency')
# plt.show()
#
# print('\n2.8--------------------------')
#
# f, ax = plt.subplots(figsize=(10,8))
# x = no_null_df['age']
# ax = sns.boxplot(x)
# ax.set_title("Visualize outliers in age variable")
# plt.show()
#
# print('\n2.9--------------------------')
#
# f, ax = plt.subplots(figsize=(10, 8))
# ax = sns.boxplot(x="income", y="age", data=no_null_df)
# ax.set_title("Visualize income wrt age variable")
# plt.show()
#
# print('\n2.10--------------------------')
#
# f, ax = plt.subplots(figsize=(10, 8))
# ax = sns.boxplot(x="income", y="age", hue="sex", data=no_null_df)
# ax.set_title("Visualize income wrt age and sex variable")
# ax.legend(loc='upper right')
# plt.show()
#
# print('\n2.11--------------------------')
#
# f, ax = plt.subplots(figsize=(10, 8))
# ax = sns.boxplot(x="race", y="age", data=no_null_df)
# ax.set_title("Visualize race wrt age variable")
# plt.show()
#
#
# print('\n2.12--------------------------')
#
# # data_frame.select_dtypes('number').corr().style.format("{:.4}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
#
# corr = data_frame.select_dtypes('number').corr()
# print(corr)
# dataplot = sns.heatmap(corr, cmap="viridis",annot=True,linewidth=0.5)
# plt.show()

print('\n2.13--------------------------')

print(data_frame)
data_frame.replace(' ?', np.NaN, inplace=True)
# data_frame.to_csv('D:/DZ/11sem/AI_Enregy/LR3/test.csv')

print('\n2.14--------------------------')

categorical = [var for var in data_frame.columns if data_frame[var].dtype == 'O']
print(data_frame[categorical].keys().values)
print(data_frame[categorical].head())

print('\n2.15--------------------------')

number = [var for var in data_frame.columns if data_frame[var].dtype != 'O']
print(data_frame[number].keys().values)
print(data_frame[number].head())

print('\n3--------------------------')

categorical.remove('income')
nocat_df = pd.get_dummies(data_frame, columns = categorical)
nocat_df['income'] = nocat_df['income'].map({' <=50K': 0, ' >50K': 1})
# nocat_df.to_csv('D:/DZ/11sem/AI_Enregy/LR3/test.csv')
print(nocat_df.head())
X = nocat_df.drop("income", axis=1)
Y = nocat_df["income"]
print(Y.head())
print(X.head())

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.33)

print('\n4--------------------------')
depth_range = np.arange(1, 50, 2)
find_optimal_param(depth_range, X_train, X_test, Y_train, Y_test, 'dtc')

print('\n5--------------------------')
find_optimal_param(depth_range, X_train, X_test, Y_train, Y_test, 'rfc')

print('\n6--------------------------')
find_optimal_param(depth_range, X_train, X_test, Y_train, Y_test, 'cbc')
