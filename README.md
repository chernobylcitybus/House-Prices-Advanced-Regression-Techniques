# House-Prices-Advanced-Regression-Techniques
*Predicting sales prices and practicing feature engineering, also testing out different regression techniques*

The ultimate goal of this project is to predict the prices of houses based on given variables.  
I set about doing this in a few steps:
## Step 1: Data Processing
First things first, I used panda's read_csv to read the train and test data, and I took a look at the data and removed unnecessary columns:  
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

>The train data size before dropping Id feature is : (1460, 81)   
>The test data size before dropping Id feature is : (1459, 80)   
>The train data size after dropping Id feature is : (1460, 80)   
>The test data size after dropping Id feature is : (1459, 79)    

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
> mu = 180932.92 and sigma = 79467.79  
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
> mu = 12.02 and sigma = 0.40  
![Graph6](https://i.imgur.com/x8Wmq8W.png)
![Graph7](https://i.imgur.com/0R0YmDb.png)  

## Step 2: Features Engineering  

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

Next I used a box-cox transform on these features:

```
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])
```
> There are 59 skewed numerical features to Box Cox transform

Then I got the dummy categorical variables:
```
all_data = pd.get_dummies(all_data)
print(all_data.shape)
```
>(2917, 220)

## Step 3: Modelling

I got the newly created train and test sets:
```
train = all_data[:ntrain]
test = all_data[ntrain:]
```
Imported the libraries I'd be needing:
```
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
```
To prevent overfitting, it is common in supervised machine learning to split the data and hold part of it as a 'test set'.  
However there is still a risk of overfitting on the test set, and so we hold out another part of the dataset to use as a 'validation set'  
We train on the training set, after which evaluation is done on the validation set, and when the experiment seems to be successful, final evaluation can be done on the test set.  
However, by splitting the data into three sets, the number of samples that can be used for learning the model is reduced, which can make the results more random. As our data set isn't massive, I decided to estimate the score via Cross Validation, as this method eliminates the need for a validation set, and so doesn't waste as much data. I used sklearn's cross_val_score function for this, and added in a line to shuffle the dataset prior to cross validation:  

```
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
```  
I then used LASSO Regression as the model. I also used sklearn's Robustscaler() method on pipeline to make the model less sensitive to the potential outliers mentioned at the beginning
```
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```
> Lasso score: 0.1115 (0.0074)  

**NB: I have since included some other models, however it seems that LASSO Regression still provides the best result**  

```
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
```
```
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```
> ElasticNet score: 0.1116 (0.0074)

```score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```
> Kernel Ridge score: 0.1153 (0.0075)

```score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```
> Gradient Boosting score: 0.1167 (0.0083)

```score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```
> Xgboost score: 0.1164 (0.0070)

```
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
```
> LGBM score: 0.1161 (0.0058)























