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
Next I removed the outliers that were mentioned by the Ames Housing Data Documentation:
```
fig, ax = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.xlabel('GrLivArea', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)

plt.show()
```
![Graph1](img/Schermata1FonZBetaZeta5.png)
