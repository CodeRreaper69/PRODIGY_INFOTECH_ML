import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score






df = pd.read_csv("house_price.csv")#dataframe object
#function for data insights
def data_insights(df):
    
    #df = pd.read_csv(d)#dataset from kaggle
    #printing first 5 rows for it
    print("FIRST 5 ROWS ARE-")
    print(df.head())
    print("\n")

    #printing the info about the dataset
    print("Information about the dataset-")
    print(df.info())
    print("\n")


    #taking important statistical insights of the dataset
    print("Some statistics of the dataset -")
    print(df.describe())
    print("\n")


    #printing the structure of the dataset
    print("ROWS,COLUMNS")
    print(df.shape)
    print("\n")

    print("columns are -")
    print(df.columns)

    print(df.columns.max())
    print(df.columns.min())

    print("Null values:")
    print(df.isna().sum())
    
#for data visualization
def data_vis(df,field1,field2,field3,field4,field5,field6,field7,field8):
    #creating excel reading object
    #df = pd.read_csv(df)#dataset from kaggle
    #print(df[field1].value_counts())
    #print(df[field2].value_counts())
    #sns.countplot(x=field1, hue=field2, data=df)
    fig, axes = plt.subplots(2,2)

    # Plotting first subplot
    axes[0,0].set_title(f"{field1} vs {field2}")
    sns.countplot(x=field1, hue=field2, data=df, ax=axes[0,0])
    
    # Plotting second subplot
    axes[0,1].set_title(f"{field3} vs {field4}")
    sns.countplot(x=field3, hue=field4, data=df, ax=axes[0,1])

    # Plotting third subplot
    axes[1,0].set_title(f"{field5} vs {field6}")
    sns.countplot(x=field3, hue=field4, data=df, ax=axes[1,0])

    # Plotting fourth subplot
    axes[1,1].set_title(f"{field7} vs {field8}")
    sns.countplot(x=field3, hue=field4, data=df, ax=axes[1,1])
    
    # enabling the plot
    plt.tight_layout()  # Adjusting layout to prevent overlapping
    plt.show()
    plt.close()


data_insights(df)
#df.dropna(inplace = True)
#data_insights(df)

#print(df['date'])

#columns of this dataset are
#Index(['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
 #      'floors', 'waterfront', 'view', 'condition', 'sqft_above',
 #      'sqft_basement', 'yr_built', 'yr_renovated', 'street', 'city',
 #      'statezip', 'country'],)



#df['total_sqft'] = df['sqft_above']+df['sqft_basement']
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_basement', 'sqft_above']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Calculate mean and standard deviation using training set
y_train_mean = y_train.mean()
y_train_std = y_train.std()

# Normalizing the target variable in the training set
y_train_normalized = (y_train - y_train_mean) / y_train_std

# Initializing and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train_normalized)

# Making predictions on the normalized test set
y_pred_normalized = model.predict(X_test)

# Reversing the normalization to get predictions in the original scale
y_pred_original_scale = y_pred_normalized * y_train_std + y_train_mean

# Calculation of MSE and R-squared on the original scale
mse = mean_squared_error(y_test, y_pred_original_scale)
r2 = r2_score(y_test, y_pred_original_scale)





#function for predicting the house price and normalizing the inputa
def predict_house_price(model, square_footage, bedrooms, bathrooms, y_train_mean, y_train_std):
    # Creating a DataFrame with user input, explicitly setting column order
    user_input_df = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [square_footage],
        'sqft_basement': [0],  # Assuming no basement in user input
        'sqft_above': [square_footage]  # Assuming all square footage is above ground
    }, columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_basement', 'sqft_above'])

    # Normalizing the user input based on training set statistics
    user_input_normalized = (user_input_df - {'sqft_living': y_train_mean, 'bedrooms': y_train_mean,
                                              'bathrooms': y_train_mean, 'sqft_basement': y_train_mean,
                                              'sqft_above': y_train_mean}) / {'sqft_living': y_train_std,
                                                                              'bedrooms': y_train_std,
                                                                              'bathrooms': y_train_std,
                                                                              'sqft_basement': y_train_std,
                                                                              'sqft_above': y_train_std}

    # Making predictions on the normalized user input
    predicted_price_normalized = model.predict(user_input_normalized)

    # Reversing the normalization to get the prediction in the original scale
    predicted_price_original_scale = predicted_price_normalized * y_train_std + y_train_mean

    return predicted_price_original_scale[0]

#main prediction part
while True:
    user_square_footage = float(input("Enter square footage: "))
    user_bedrooms = int(input("Enter number of bedrooms: "))
    user_bathrooms = float(input("Enter number of bathrooms: "))

    predicted_price = predict_house_price(model, user_square_footage, user_bedrooms, user_bathrooms, y_train_mean, y_train_std)

    print(f"The predicted price for the given input is: ${predicted_price:,.2f}")
    n = input("Continue again?(y/n):")
    if n in ['n','N']:
        break
    else:
        pass
    


print("\n")

































