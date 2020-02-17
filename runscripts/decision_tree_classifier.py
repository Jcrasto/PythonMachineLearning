import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def one_hot_encode(df,encode_cols):
    one_hot = pandas.get_dummies(df[encode_cols])
    df = pandas.concat((df,one_hot), axis=1)
    col_names = df.columns.values
    df.reindex(columns=col_names)
    return(df)

def create_decision_tree_model(df,numeric_cols):
    X = df[numeric_cols]
    y = df['label']
    train_X, val_X, train_y, val_y = train_test_split(X,y)

    model = DecisionTreeClassifier(max_leaf_nodes=500, max_depth=100)
    model.fit(train_X,train_y)

    predictions = model.predict(val_X)
    eval = pandas.DataFrame()
    eval['actual'] = val_y
    eval['predicted'] = predictions
    eval['check'] = eval['actual'] == eval['predicted']
    # return(model)
    return(eval)

def final_model_test(model,col_names,numeric_cols,encode_cols):
    test_df = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", skiprows=1, header=None,
                         names=col_names, index_col=False)
    test_df = one_hot_encode(test_df,encode_cols)
    col_names = test_df.columns.values
    test_df.reindex(columns=col_names)
    X = test_df[numeric_cols]
    y = test_df['label']
    predictions = model.predict(X)
    eval = pandas.DataFrame()
    eval['actual'] = y
    eval['predicted'] = predictions
    eval['check'] = eval['actual'] == eval['predicted']
    return(eval)

if __name__ == "__main__":
    col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                 "relationship","race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]
    df = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None,
                         names=col_names, index_col=False)
    numeric_cols = df.describe().columns.values
    encode_cols = ['sex','race','workclass','marital-status','native-country','occupation']
    df = one_hot_encode(df,encode_cols)
    numeric_cols = df.describe().columns.values
    eval = create_decision_tree_model(df,numeric_cols)
    print(str(eval['check'].sum() / eval.shape[0]) + "% predicted correctly")