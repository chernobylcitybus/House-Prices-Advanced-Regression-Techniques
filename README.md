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









