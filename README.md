# House-Prices-Advanced-Regression-Techniques
*Predicting sales prices and practicing feature engineering, also testing out different regression techniques*

The ultimate goal of this project is to predict the prices of houses based on given variables.  
I set about doing this in a few steps:
### Step 1: Data Processing
First things first, I took a look at the data and removed unnecessary columns:  
```
train = pd.read_csv('C:/Users/Luke/Downloads/train.csv')  
test = pd.read_csv('C:/Users/Luke/Downloads/test.csv') 

train.head(5)  
test.head(5)    

#check the numbers of samples and features  
print("The train data size before dropping Id feature is : {} ".format(train.shape))  
print("The test data size before dropping Id feature is : {} ".format(test.shape))  

# Save the 'Id' column  
train_ID = train['Id']  
test_ID = test['Id']  

# Now drop the 'Id' column since it's unnecessary for  the prediction process.  
train.drop("Id", axis=1, inplace=True)  
test.drop("Id", axis=1, inplace=True)  

# Checking the data size again after dropping the 'Id' variable  
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))  
print("The test data size after dropping Id feature is : {} ".format(test.shape))
```
Next I took a look at the outliers that were mentioned by the Ames Housing Data Documentation:
```
fig, ax = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.xlabel('GrLivArea', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)

plt.show()
```
![Graph1](https://i.imgur.com/AKD6FgK.png)  
The documentation recommended deleting all data with GrLivArea > 4000, however the upper 2 'outliers' fit in, whereas the bottom 2 definitely dont.  
I didn't want to be excessive with the outlier removal as it could negatively impact the models if there are also outliers in the test data, so I just chose to remove the egregious ones:
```# Deleting outliers
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# Check the graph again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.xlabel('GrLivArea', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)
plt.show()
```
![Graph2](https://i.imgur.com/UN4865e.png)

Next I checked the relationship between the 'main' features in a multiplot format to get a better overall feel for the data. I found these features with the correlation matrix shown a bit later:
```#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols])
plt.show()
```
![Graph3](https://i.imgur.com/vomRVPq.png)  

SalePrice is the target variable that I'm going to try to predict, so I did some analysis on it, getting the mu and sigma values and finding the distribution and probability plots:
```
# SalePrice target variable analysis ------------

sns.distplot(train['SalePrice'] , fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
```
![Graph4](https://i.imgur.com/u9jbNlx.png)

```
#Get the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
```
![Graph5](https://i.imgur.com/bwCPfUo.png) 

This shows that the SalePrice variable is right skewed. I'll need to transform it to a normal distribution for the models to work properly:

```
#Use the numpy fuction log1p to  apply log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Plot the new distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get the new QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
```
![Graph6](https://i.imgur.com/x8Wmq8W.png)
![Graph7](https://i.imgur.com/0R0YmDb.png)

##Step 2: Features Engineering  

Now I linked the train and test data together: (**NB: this is a mistake that allows for data-leakage, will be fixed in the future**)
```
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
```
>all_data size is : (2917, 79)

Next I found all the missing data:
```
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Percentage' :all_data_na})
missing_data.head(20)
```
Id | Missing Percentage
------------ | -------------
PoolQC	| 99.691464
MiscFeature	| 96.400411
Alley	| 93.212204
Fence	| 80.425094
FireplaceQu	| 48.680151
LotFrontage	| 16.660953
GarageFinish | 5.450806
GarageYrBlt	| 5.450806
GarageQual	| 5.450806
GarageCond	| 5.450806
GarageType	| 5.382242
BsmtExposure | 2.811107
BsmtCond	| 2.811107
BsmtQual	| 2.776826
BsmtFinType2 | 2.742544
BsmtFinType1 | 2.708262
MasVnrType	| 0.822763
MasVnrArea	| 0.788481
MSZoning	| 0.137127
BsmtFullBath | 0.068564

And graphed it for fun:
```
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
```
![Graph9](https://i.imgur.com/CjqJt79.png)

I also  got a correlation matrix  to see which features were most strongly related to SalePrice and to each other:
```
#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
```
![Graph10](https://i.imgur.com/TJCVW2E.png)
```
# saleprice correlation matrix
k = 10  # number of heatmap variables
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()
```
![Graph11](https://i.imgur.com/K90vcGu.png)

Next it was time to fill in all the missing values! I went through all the data and filled in the appropriate values. (For example PoolQC had a 99% 'missing data' percentage. This just meant that those homes didn't have a pool, and so I changed the 'missing' value to "None").
```
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
```
For the LotFrontage missing values I made the assumption that most houses in a neighbourhood would have a similar size, so I filled in the missing values by taking the median of the LotFrontage values of houses in the same neighbourhood:
```
#Group by neighbourhood and fill in missing value with the median of the LotFrontage for the respective neighbourhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
```
For the 'Garage', 'Basement', and 'Masonry Veneer' categorical variables, the missing data indicated no garage/basement/msnvnr, so I filled the missing values with "None". Then I filled the numerical values with 0:
```
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')  
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)  
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)  
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)  
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
```
The vast majority of houses were 'Residential Low Density', so I filled in the missing values with 'RL':
```
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
```
All records for 'Utilities' are "AllPub", except for one ("NoSeWa") and 2 missing values. The house with "NoSeWa" is in the training set, meaning this feature wont help with predictive modelling, so I chose to just drop it.
```
all_data = all_data.drop(['Utilities'], axis=1)
```
The next features had only a few missing values, so I replaced them with the feature's mode:
```
all_data["Functional"] = all_data["Functional"].fillna("Typ")  
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])  
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])  
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])  
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])  
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])  
```
Finally done! Now to check if I left anything out:
```
#Check for remaining missing values 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()
```

_ | Missing Ratio
--- | ---
_ | _


    


Next I changed some numerical values that we're actually categorical values, and applied sklearn's LabelEncoder to them, to gain information from them:
```
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
```
> Shape all_data: (2917, 78)  
I also added a new feature showing the total square footage, as this seems very important to house pricing: **NB: In the future I will also add a few more features as well as combine some of the 'Garage' and 'Basement' features as some of them depict the exact same information**
```
# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
```
Next I needed to find the skewness and then transform all numerical features:
```
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
```
_ | Skew
--- | ---
MiscVal	| 21.939672
PoolArea	| 17.688664
LotArea	| 13.109495
LowQualFinSF | 12.084539
3SsnPorch	| 11.372080
LandSlope	| 4.973254
KitchenAbvGr | 4.300550
BsmtFinSF2	| 4.144503
EnclosedPorch | 4.002344
ScreenPorch	| 3.945101



















