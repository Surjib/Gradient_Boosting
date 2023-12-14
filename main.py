import string

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras.datasets import mnist
from keras.src.layers import Dense, Activation, Dropout
from keras.src.utils import np_utils
from sklearn import model_selection
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
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

        f_score_test.append(f1_score(Y_test, Y_predict, average= 'micro'))
        f_score_train.append(f1_score(Y_train, Y_predict_train, average= 'micro'))

    matrix = np.matrix(np.c_[param_range, f_score_test, f_score_train])
    models = pd.DataFrame(data=matrix,
                          columns=[param, 'test F-Measure', 'train F-Measure'])

    best_index = models['test F-Measure'].idxmax()
    best_value = models.iloc[best_index]['test F-Measure']

    print('Best F-measure for {} using depth value {} index {} '.format(model_name, best_value, best_index))

    plt.subplot(1, 2, 1)
    plt.plot(param_range, f_score_test)
    plt.plot(param_range[best_index], f_score_test[best_index], marker='o', color='red')
    plt.annotate(xy=(param_range[best_index], f_score_test[best_index]),
                 text='({}, {})'.format(("%.1f" % f_score_test[best_index]), ("%.1f" % param_range[best_index])))
    plt.title("Test F_Measure", fontsize=10)
    plt.ylabel('Accuracy score(%)', fontsize=8)
    plt.xlabel(param, fontsize=8)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(param_range, f_score_train)
    plt.plot(param_range[best_index], f_score_train[best_index], marker='o', color='red')
    plt.annotate(xy=(param_range[best_index], f_score_train[best_index]),
                 text='({}, {})'.format(("%.1f" % f_score_train[best_index]), ("%.1f" % param_range[best_index])))
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
print('Признаки, в которых есть пропущенные значенмя {}\n'.format(null_vals.index.values))

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
dataplot = sns.heatmap(corr, cmap="viridis",annot=True,linewidth=0.5)
plt.show()

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

print('\n7.1--------------------------')

# print(data_frame.isna().sum()) # есть NaN

data_frame['workclass'].fillna(data_frame['workclass'].mode()[0], inplace=True)
data_frame['occupation'].fillna(data_frame['occupation'].mode()[0], inplace=True)
data_frame['native_country'].fillna(data_frame['native_country'].mode()[0], inplace=True)

# print(data_frame.isna().sum()) # нет NaN


Y = data_frame["income"].map({' <=50K': 0, ' >50K': 1})
X = data_frame.drop("income", axis=1)
X = pd.get_dummies(X)

# print(Y.head())
# print(X.head())

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.33)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# X_train = X_train.to_numpy(dtype=('float32'))
# X_test = X_test.to_numpy(dtype=('float32'))

Y_train = np_utils.to_categorical(Y_train, 2)
Y_test = np_utils.to_categorical(Y_test, 2)

print(X_train)
print(X_test)
print(Y_test)
print(Y_train)

print('\n7.2--------------------------')

NB_CLASSES = Y_train.shape[1]
INPUT_SHAPE = (X_train.shape[1],)
model = Sequential()
model.add(Dense(32, input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

EPOCHS = 30
epoch_range = np.arange(1,31,1)
history = model.fit(X_train, Y_train, batch_size = 32, epochs = EPOCHS, verbose = 1, validation_data = (X_test, Y_test))

f1_score_list_train = []
f1_score_list_test = []
for i in range(EPOCHS):
    # a = history.history()
    # print(a)
    f1_score_list_train.append(2 * history.history['precision'][i] *
    history.history['recall'][i] / (history.history['precision'][i] +
    history.history['recall'][i]))
#
    f1_score_list_test.append(2 * history.history['val_precision'][i] *
    history.history['val_recall'][i] / (history.history['val_precision'][i] +
    history.history['val_recall'][i]))

# print(len(f1_score_list_test))
# print(f1_score_list_train)

matrix = np.matrix(np.c_[epoch_range, f1_score_list_test, f1_score_list_train])
models = pd.DataFrame(data=matrix,
                      columns=['epoch', 'test F-Measure', 'train F-Measure'])

best_index = models['test F-Measure'].idxmax()
best_value = models.iloc[best_index]['test F-Measure']

plt.subplot(1, 2, 1)
plt.plot(epoch_range, f1_score_list_test)
#
plt.plot(epoch_range[best_index], f1_score_list_test[best_index], marker='o', color='red')
plt.annotate(xy=(epoch_range[best_index], f1_score_list_test[best_index]),
             text='({}, {})'.format(("%.1f" % best_value), ("%.1f" % epoch_range[best_index])))
plt.title("Test F_Measure", fontsize=10)
plt.ylabel('Accuracy score(%)', fontsize=8)
plt.xlabel('Epoch', fontsize=8)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epoch_range, f1_score_list_train)
plt.annotate(xy=(epoch_range[best_index], f1_score_list_train[best_index]),
             text='({}, {})'.format(("%.1f" % f1_score_list_train[best_index]), ("%.1f" % epoch_range[best_index])))
plt.title("Train F_Measure", fontsize=10)
plt.ylabel('Accuracy score(%)', fontsize=8)
plt.xlabel('Epoch', fontsize=8)
plt.grid(True)

plt.show()



y_prediction = model.predict(X_test)
y_prediction = np.argmax (y_prediction, axis = 1)
y_test=np.argmax(Y_test, axis=1)

result = confusion_matrix(y_test, y_prediction , normalize='pred')
conf_matrix = pd.DataFrame(data=result, columns=['predicted ham', 'predicted spam'],
                           index=['actual ham', 'actual spam'])

print(conf_matrix)

print('\n8--------------------------')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# pick a sample to plot
sample = 1
image = X_train[sample]
# plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()

X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)

# print(X_train)
# print(X_test)
# print(y_test)
# print(y_train)

find_optimal_param(epoch_range, X_train, X_test, y_train, y_test, 'cbc')
