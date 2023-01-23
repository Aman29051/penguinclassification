import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('penguins_cleaned.csv')
df = data.copy()

target = 'species'
encode = ['sex','island']

# Converting object type features and target into int or float
for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
df['species'] = df['species'].map(target_mapper)

# Separating dependent and independent features
X = df.drop('species',axis=1)
y = df['species']

model = RandomForestClassifier()
model.fit(X,y)

pickle.dump(model,open('model.pkl','wb'))



