import pandas
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
import pandas
import time
from dask_ml.model_selection import train_test_split as dask_split


def dask_run():
    # df = dd.read_csv(...)
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    df = dd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )
    df = df.rename(columns=dict(zip(df.columns, col_names)))
    numeric_cols = df.describe().columns.values
    X = df[numeric_cols]
    y = df["label"]
    train_X, val_X, train_y, val_y = dask_split(X, y)
    return (
        train_X,
        val_X,
        train_y,
        val_y,
    )  # This uses the single-machine scheduler by default


def pandas_run():
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    df = pandas.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None,
        names=col_names,
        index_col=False,
    )
    numeric_cols = df.describe().columns.values
    X = df[numeric_cols]
    y = df["label"]
    train_X, val_X, train_y, val_y = train_test_split(X, y)
    return (train_X, val_X, train_y, val_y)


def create_rf(train_X, val_X, train_y, val_y):
    start = time.time()
    model = RandomForestClassifier()
    model.fit(train_X, train_y)

    predictions = model.predict(val_X)
    print(
        str(type(train_X))
        + str(accuracy_score(val_y, predictions) * 100)
        + "% predicted correctly"
    )
    end = time.time()
    print(str(type(train_X)) + " :took " + str(end - start))
    return ()


if __name__ == "__main__":
    train_X, val_X, train_y, val_y = dask_run()
    create_rf(train_X, val_X, train_y, val_y)
    train_X, val_X, train_y, val_y = pandas_run()
    create_rf(train_X, val_X, train_y, val_y)
