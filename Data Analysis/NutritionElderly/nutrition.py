import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
xls = 'http://www.biostatisticien.eu/springeR/nutrition_elderly.xls'
nutri = pd.read_excel(xls)
pd.set_option('display.max_columns', 8)
#print(nutri.head(3))
#print(nutri.info()) #allows us to check the type/structure of variables
#we can modify Python value and type for each categorical feature, using replace
# and astype methods. 
DICT = {1 : 'Male', 2 : 'Female'} #dictionary that specifies replacement
nutri['gender'] = nutri['gender'].replace(DICT).astype('category')
#replace 1 with Male and 2 with Female
nutri['height'] = nutri['height'].astype(float)
#nutri.to_csv('nutri.csv', index = False)

nutri = pd.read_csv('nutri.csv')
#print(nutri)
FAT_MAP = {1 : 'butter',
           2 : 'margarine',
           3 : 'peanut',
           4 : 'sunflower',
           5 : 'olive',
           6 : 'Isio4',
           7 : 'colza',
           8 : 'duck'}
nutri['fat'] = nutri['fat'].replace(FAT_MAP).astype('category')
print(nutri['fat'].describe())
print(nutri['fat'].value_counts())

SITUATION_MAP = {1 : 'Single',
                 2 : 'Couple', 
                 3: 'Family'}

#Cross tabulation
nutri['situation'] = nutri['situation'].replace(SITUATION_MAP).astype('category')
print(pd.crosstab(nutri.gender, nutri.situation, margins = True))
#margins = True adds a row and column totals.

#summary stats
nutri['height'].mean()
nutri['height'].quantile(q = [0.25, 0.5, 0.75])
round(nutri['height'].var(), 2) #rounds to 2 decimal places
round(nutri['height'].std(), 2)
nutri["height"].describe()


#visualizing data
#bar plot
"""
width = 0.35 #width of bars
x = [0, 0.8, 1.6] #bar positions on x axis
situation_counts = nutri['situation'].value_counts()
plt.bar(x, situation_counts, width, edgecolor = 'black')
plt.xticks(x, situation_counts.index) #puts labels on x axis
plt.show()

"""
#box plot
plt.boxplot(nutri['age'], widths = 0.35, vert = False)
plt.xlabel('age')
plt.show()