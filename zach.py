#!/usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

df = pd.read_csv('Programmatic Project_Scoring_TTD pixel fires.csv', encoding='latin1')


Y = (df.trackingtagid.as_matrix() == 'qelg9wq')



#                   Remove categorical columns with too many unique values
#categorical_columns = ['country', 'region', 'metro', 'organizationname', 'devicetype', 'osfamily', 'browser', 'devicemake', 'devicemodel']
categorical_columns = ['country', 'region', 'metro', 'devicetype', 'osfamily', 'browser', 'devicemake']
#{c: np.unique(df[c].as_matrix()).shape[0] for c in categorical_columns}
#sum({c: np.unique(df[c].as_matrix()).shape[0] for c in categorical_columns}.values())


X = pd.get_dummies(df[categorical_columns]).as_matrix()


#nb = MultinomialNB()
nb = BernoulliNB()


nb.fit(X, Y)
probs = nb.predict_log_proba(X)


result = df.sort_values('score', ascending=False).drop_duplicates('tdid', keep='first')[:10000]
result[['tdid','score']].to_csv('zach_output.csv', index=False)
