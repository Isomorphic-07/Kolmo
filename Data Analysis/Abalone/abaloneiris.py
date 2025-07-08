import pandas as pd
abalone = pd.read_csv('Data Analysis/Abalone/abalone.data', header = None)
#header = None is important to let Python know that the first line of the CSV does
# not contain the names of the features. 
urlprefix = 'https://vincentarelbundock.github.io/Rdatasets/csv/'
dataname = 'datasets/iris.csv'
iris = pd.read_csv(urlprefix + dataname)

iris = iris.drop('rownames', axis = 1)

print(iris.head())
#.head() is a datafram method that gives the first few rows of the DataFram, including
# feature names, number of rows can be passed as an argument and 5 is by default. 
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                   'Viscera weight', 'Shell weight', 'Rings']
print(abalone.head(3))