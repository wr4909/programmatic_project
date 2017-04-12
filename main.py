#!/usr/bin/python3

import numpy as np
import pandas as pd
import datetime
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

df = pd.read_csv('Programmatic Project_Scoring_TTD pixel fires.csv', encoding='latin1', parse_dates=['logentrytime'])

# Remove duplicate tdid
visited_tdids = df[df.trackingtagid.as_matrix() == 'qelg9wq']['tdid']
df['contact'] = df.tdid.isin(visited_tdids)
df = df.sort_values('logentrytime', ascending=False).drop_duplicates('tdid', keep='first')

Y = df['contact'].as_matrix()



#                   Remove categorical columns with too many unique values
#categorical_columns = ['country', 'region', 'metro', 'organizationname', 'devicetype', 'osfamily', 'browser', 'devicemake', 'devicemodel']
categorical_columns = ['country', 'region', 'metro', 'devicetype', 'osfamily', 'browser', 'devicemake']
#{c: np.unique(df[c].as_matrix()).shape[0] for c in categorical_columns}
#sum({c: np.unique(df[c].as_matrix()).shape[0] for c in categorical_columns}.values())


X = pd.get_dummies(df[categorical_columns]).as_matrix()


#nb = MultinomialNB()
nb = BernoulliNB()


nb.fit(X, Y)
probs = nb.predict_proba(X)
df['score'] = probs[:,1]

daterange = datetime.date.today() - datetime.timedelta(days=90)
result = df[df.logentrytime >= daterange].sort_values('score', ascending=False)[:10000]
result[['tdid','score']].to_csv('output.csv', index=False)
