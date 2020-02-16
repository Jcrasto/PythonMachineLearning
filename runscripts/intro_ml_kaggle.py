import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]
df = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",header=None, names=col_names, index_col=False)
print(df.describe())

X = df[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']]
y = df[col_names[-1]]
train_X, val_X, train_y, val_y = train_test_split(X,y)

model = DecisionTreeClassifier()
model.fit(train_X,train_y)

predictions = model.predict(val_X)
eval = pandas.DataFrame()
eval['actual'] = val_y
eval['predicted'] = predictions
eval['check'] = eval['actual'] == eval['predicted']
print(str(eval['check'].sum()/eval.shape[0]) + "% predicted correctly")
#mean

#test_df = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", header=None, skiprows=1)
#print(test_df.head())