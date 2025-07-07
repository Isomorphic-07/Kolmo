import pandas as pd
abalone = pd.read_csv('Data Analysis/abalone.data', header = None)
#header = None is important to let Python know that the first line of the CSV does
# not contain the names of the features. 
urlprefix = 'https://vincentarelbundock.github.io/Rdatasets/csv/'
dataname = 'datasets/iris.csv'
iris = pd.read_csv(urlprefix + dataname)
print(iris.head())