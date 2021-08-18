import pandas as pd
from sys import exit
from numpy import array
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from category_encoders import TargetEncoder

developmentRawAddress = './Dataset/development.tsv'
testRawAddress = './Dataset/evaluation.tsv'
df = pd.read_csv(developmentRawAddress, sep='\t')
dfTest = pd.read_csv(testRawAddress, sep='\t')

# for i in df.columns:
#     print("column: ", i, "   - Number of Null: ", df[i].isnull().sum(), "   - Number of Uniques: ",
#           len(pd.unique(df[i])), "\n")

# for i in dftest.columns:
#     print("column: ", i, "   - Number of Null: ", dftest[i].isnull().sum(), "   - Number of Uniques: ",
#           len(pd.unique(dftest[i])))
# exit()
clmns = ['user/gender', 'user/birthdayUnix', 'user/birthdayRaw', 'user/ageInSeconds', 'user/profileName']
rows = ['review/text']

df.drop(clmns, axis=1, inplace=True)
df.dropna(subset=rows, inplace=True)
dfTest.drop(clmns, axis=1, inplace=True)

#####################################################################################
imputer = KNNImputer(n_neighbors=5)
dfNew = df[['beer/ABV', 'review/appearance', 'review/aroma', 'review/palate', 'review/taste']]
imputer.fit(dfNew)
TempData = imputer.transform(dfNew)
dfNew = pd.DataFrame(data=TempData,
                     columns=['beer/ABV', 'review/appearance', 'review/aroma', 'review/palate', 'review/taste'])
dfNew['beer/ABV'] = dfNew['beer/ABV'].apply(lambda x: round(x, 1))
df['beer/ABV'] = dfNew['beer/ABV']

dfTestNew = dfTest[['beer/ABV', 'review/appearance', 'review/aroma', 'review/palate', 'review/taste']]
# imputer.fit(dfTestNew)
dfTestNew = pd.DataFrame(data=imputer.transform(dfTestNew),
                         columns=['beer/ABV', 'review/appearance', 'review/aroma', 'review/palate', 'review/taste'])

dfTest["review/text"].fillna("the", inplace=True)
dfTestNew['beer/ABV'] = dfTestNew['beer/ABV'].apply(lambda x: round(x, 1))
dfTest['beer/ABV'] = dfTestNew['beer/ABV']

Encoder = TargetEncoder()

beerName = Encoder.fit_transform(df['beer/name'][:].values, df['beer/ABV'][:].values)
# beerStyle = Encoder.fit_transform(df['beer/style'][:].values, df['beer/ABV'][:].values)
beerNameTest = Encoder.fit_transform(dfTest['beer/name'][:].values, dfTest['beer/ABV'][:].values)
# beerStyleTest = Encoder.fit_transform(dfTest['beer/style'][:].values, dfTest['beer/ABV'][:].values)
# replace these columns with the one in the origin dataset and testset

scaler = MinMaxScaler(feature_range=(1, 5))
scaler.fit(beerName)
beerName = scaler.transform(beerName)
beerNameTest = scaler.transform(beerNameTest)
tmp = df['beer/ABV'].values.reshape(-1, 1)
df['beer/ABV'] = scaler.fit_transform(tmp)
tmp = dfTest['beer/ABV'].values.reshape(-1, 1)
dfTest['beer/ABV'] = scaler.transform(tmp)

df['beer/name'] = beerName  # [0:66861]
dfTest['beer/name'] = beerNameTest  # [66861:96861]

beerStyle = df['beer/style'].values.tolist()
styleArray = array(beerStyle)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(styleArray)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
df1 = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoder.get_feature_names())
df.drop('beer/name', axis=1, inplace=True)
dfNew = pd.concat([df, df1], axis=1)

beerStyleTest = dfTest['beer/style'].values.tolist()
styleArray = array(beerStyleTest)
integer_encoded = label_encoder.transform(styleArray)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.transform(integer_encoded)
df1Test = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoder.get_feature_names())
dfTest.drop('beer/name', axis=1, inplace=True)
dfTestNew = pd.concat([dfTest, df1Test], axis=1)

