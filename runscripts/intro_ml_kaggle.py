import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]
df = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",header=None, names=col_names, index_col=False)
numeric_cols = df.describe().columns.values

one_hot = pandas.get_dummies(df[['sex','race','workclass','marital-status','native-country','occupation']])
numeric_cols = np.append(numeric_cols,one_hot.columns.values)
df = pandas.concat((df,one_hot), axis=1)
# col_names = df.columns.values
df.reindex(columns=col_names)

X = df[numeric_cols]
y = df['label']
train_X, val_X, train_y, val_y = train_test_split(X,y)

model = DecisionTreeClassifier()
model.fit(train_X,train_y)

predictions = model.predict(val_X)
eval = pandas.DataFrame()
eval['actual'] = val_y
eval['predicted'] = predictions
eval['check'] = eval['actual'] == eval['predicted']
print(str(eval['check'].sum()/eval.shape[0]) + "% predicted correctly")
# mean_absolute_error(predictions, val_y)
#
# test_df = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", header=None, skiprows=1)
# print(test_df.head())