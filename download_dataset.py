# download_dataset.py
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_data = pd.read_csv(url, header=None, names=col_names)

iris_data.to_csv("data/iris.csv", index=False)
