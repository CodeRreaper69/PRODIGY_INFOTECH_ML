import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler




df = pd.read_csv("Mall_Customers.csv")#dataframe object
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
def data_vis1(df,field1,field2,field3,field4,field5,field6,field7,field8):
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
df.dropna(inplace = True)
data_insights(df)

# Extracting the relevant features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determining the optimal number of clusters using the elbow method to find the optimal value of k and hence the calculating the sum of squared distances from each point to its assigned cluster centre
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)




# Displaying the clusters or the segmentation
print(df[['CustomerID', 'Cluster']])

def data_vis(df,field1,field2):
    #fig, axes = plt.subplots(1,2)
    #plt.title(f"{field1} vs {field2}")
    sns.barplot(x=field2, hue=field2, data=df)
    plt.tight_layout()  # Adjusting layout to prevent overlapping
    plt.show()
    plt.close()

data_vis(df,df['CustomerID'], df['Cluster'])





