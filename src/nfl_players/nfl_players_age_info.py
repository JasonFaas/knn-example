import pandas as pd
import numpy as np
import math
import operator







df = pd.read_csv('Basic_Stats.csv')
columns = ['Birth Place', 'Birthday', 'Height (inches)', 'Weight (lbs)', 'Current Status', 'Years Played']
df = df[columns]

df = df.dropna()

df = df.rename(columns={"Birth Place": "birth_place", "Current Status":"current_status", "Years Played" : "years_played", "Birthday":"birthday"})
# df = df.drop(df[df['current_status'] != 'Retired'].index)

from datetime import datetime

def birthmonth(row):
    return datetime.strptime(row['birthday'], '%m/%d/%Y')
    # return row['birthday'][0:2]
df['birth_day_dt'] = df.apply(birthmonth, axis = 1)
def birthyear(row):
    return row['birth_day_dt'].year
def birthmonth(row):
    return row['birth_day_dt'].month
df['birth_year'] = df.apply(birthyear, axis=1)
df['birth_month'] = df.apply(birthmonth, axis=1)

# print(df.size)
df = df.drop(df[df['birth_place'].str.contains(",") == False].index)
def set_birth_state(row):
    birth_place = row['birth_place']
    return birth_place.split(',')[1].strip()
df['birth_state'] = df.apply(set_birth_state, axis = 1)

def set_years_played(row):
    split = row['years_played'].split('-')
    return (int(split[1].strip()) - int(split[0].strip()) + 1)

df['years_played_count'] = df.apply(set_years_played, axis = 1)

# print(df.head())
# print(df.columns.values)
# print(df)

columns_for_fa = ['Height (inches)',
                  'Weight (lbs)',
                  'birth_year',
                  'years_played_count']

df_f_fa = df[columns_for_fa]

from sklearn import decomposition, preprocessing

# df_f_fa_n = df_f_fa
df_f_fa_n = preprocessing.scale(df_f_fa) # data normalisation attempt

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(df_f_fa_n)
# print(df[['birth_state']])
# print(len(principal_components))

pc_ = ['pc1', 'pc2', 'pc3']
principal_df = pd.DataFrame(data=principal_components, columns=pc_)
birth_state_df = pd.DataFrame(data=df[['birth_state']].values, columns=['birth_state'])
final_df = pd.concat(objs=[principal_df, birth_state_df], axis=1)
# print(final_df)

# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_df[pc_].values,
                                                    final_df[['birth_state']].values,
                                                    train_size=0.8,
                                                    random_state=42)


## Import the Classifier.
from sklearn.neighbors import KNeighborsClassifier

train_test_df = pd.DataFrame(columns=['knn', 'percent'])
# train_test_df = train_test_df.append({'knn': 0, 'percent':0}, ignore_index=True)





for n_neighbors_value in range(1,100):
    ## Instantiate the model with 5 neighbors.
    knn = KNeighborsClassifier(n_neighbors=n_neighbors_value)
    ## Fit the model on the training data.
    knn.fit(X_train, np.array(y_train)[:,0])
    ## See how the model performs on the test data.
    score = knn.score(X_test, np.array(y_test)[:, 0])
    train_test_df = train_test_df.append({'knn': n_neighbors_value, 'percent':score}, ignore_index=True)

# print(train_test_df)
print(train_test_df['percent'].idxmax())
print()
print(train_test_df.loc[train_test_df['percent'].idxmax()])
