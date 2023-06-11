import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BLR import BayesianLinearRegression
from util import corrMat, snsPairGrid, modelEval


df = pd.read_csv('data/housing.csv')
print(df.info())

# remove rows with missing values
df = df.dropna()
df = df.drop('ocean_proximity', axis=1) # drop categorical variable

# separate training and test data
df_train = df.sample(frac=0.8, random_state=0)
df_test = df.drop(df_train.index)

df_train.hist(bins=60, figsize=(15,9), color='darkviolet')
plt.show()

corrMat(df_train)
snsPairGrid(df_train)

print("Before changing")
print("With scaling")
modelEval(df_train, df_test, scaling_id=True)
print("Without scaling")
modelEval(df_train, df_test, scaling_id=False)


maxval2 = df_train['median_house_value'].max() # get the maximum value
trdata_upd = df_train[df_train['median_house_value'] != maxval2]
tedata_upd = df_test[df_test['median_house_value'] != maxval2]
trdata_upd.hist(bins=60, figsize=(15,9),color="darkviolet");plt.show() # looks like its completely removed.

# Make a feature that contains both longtitude & latitude
trdata_upd['diag_coord'] = (trdata_upd['longitude'] + trdata_upd['latitude'])         # 'diagonal coordinate', works for this coord
trdata_upd['bedperroom'] = trdata_upd['total_bedrooms'] / trdata_upd['total_rooms']
trdata_upd.drop('total_bedrooms', axis=1, inplace=True) # drop the original feature
trdata_upd.drop('total_rooms', axis=1, inplace=True) # drop the original feature
trdata_upd.drop('longitude', axis=1, inplace=True) # drop the original feature
trdata_upd.drop('latitude', axis=1, inplace=True) # drop the original feature

# update test data as well
tedata_upd['diag_coord'] = (tedata_upd['longitude'] + tedata_upd['latitude'])
tedata_upd['bedperroom'] = tedata_upd['total_bedrooms']/tedata_upd['total_rooms']     # feature w/ bedrooms/room ratio
tedata_upd.drop('total_bedrooms', axis=1, inplace=True) # drop the original feature
tedata_upd.drop('total_rooms', axis=1, inplace=True) # drop the original feature
tedata_upd.drop('longitude', axis=1, inplace=True) # drop the original feature
tedata_upd.drop('latitude', axis=1, inplace=True) # drop the original feature


corrMat(trdata_upd)
snsPairGrid(trdata_upd)


print("After changing")
print("With scaling")
modelEval(trdata_upd, tedata_upd, scaling_id=True)
print("Without scaling")
modelEval(trdata_upd, tedata_upd, scaling_id=False)


