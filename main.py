#!/usr/bin/env python
# coding: utf-8

# ## SALES FORECASTING SYSTEM

# ### EDA

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('dataset.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[9]:


df.columns


# In[10]:


df.isnull().sum()


# In[19]:


columns = ['Unit price','Total','Rating','gross income']

# Plot the distribution of the data

for col in columns:
    sns.histplot(df[col],kde=True)
    
    # Fit a normal distribution to the data
    
    (mu,sigma) = stats.norm.fit(df[col])
    print('{} : mu = {:.2f},sigma={:.2f}'.format(col,mu,sigma))
    
    # Calculate the skewness and kurtosis of the data
    
    print('{} : Skewness: {:.2f}'.format(col,df[col].skew()))
    print('{} : Kurtosis: {:.2f}'.format(col,df[col].kurt()))
    
    # Add the fitted normal distribution to the plot
    
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 180)
    y = stats.norm.pdf(x,mu,sigma)
    plt.plot(x,y,label = 'Normal Fit')
    
    # Add labels and title to the plot
    
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title('Distribution of {}'.format(col))
    
    # Plot the QQ-plot
    
    fig = plt.figure()
    stats.probplot(df[col],plot=plt)
    
    plt.show()


# #### Unit Price Distribution

# In[21]:


# Box plots and Swarm plots

columns = ['Unit price','Total','Rating','gross income']

plt.figure(figsize=(20,20))

for i, column in enumerate(columns):
    plt.subplot(len(columns),2,i+1)
    sns.boxplot(x='Product line',y=column,data=df,palette='coolwarm')
    sns.swarmplot(x='Product line',y=column,data=df,color='black',alpha=0.4)
    plt.title(f'{column} by Product Line')

plt.tight_layout()
plt.show()


# #### Transaction Density during open hours

# In[22]:


df['Time'] = pd.to_datetime(df['Time'])
df = df[(df['Time'].dt.hour >=10) & (df['Time'].dt.hour < 21)]


# In[24]:


df['MinutesFromOpening'] = (df['Time'].dt.hour -10) * 60 + df['Time'].dt.minute

# Create a distribution plot to visualize the transaction density

plt.figure(figsize=(20,10))
sns.histplot(data = df, x = 'MinutesFromOpening', bins=60, kde = True)
plt.title('Transaction Density Throughout Store Open Hours', fontsize = 20)
plt.xlabel('Minutes From Opening (10:00)', fontsize = 16)
plt.ylabel('Number of Transactions', fontsize = 16)
plt.xticks(np.arange(0,660,60),[f"{10+t//60:02d}:{t%60:02d}" for t in np.arange(0,660,60)])
plt.grid(True)
plt.show()


# #### Check Distribution

# In[27]:


plt.figure(figsize = (12,6))
sns.barplot(x = 'Product line', y = 'Total', hue = 'Gender', data = df)
plt.xticks(rotation = 45)
plt.show()


# In[30]:


g = sns.FacetGrid(df, col = 'Payment', row = 'Customer type', hue = 'Gender', margin_titles = True)
g.map(sns.barplot, 'Product line','Total')

g.add_legend(title='Gender', bbox_to_anchor=(1.05,0.5), loc='center left', borderaxespad = 0)

for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 45, ha = 'right', fontsize = 10)
    
plt.show()


# #### Weekly Sales by City

# In[31]:


df['Date'] = pd.to_datetime(df['Date'])
weekly_data = df.groupby(['City',pd.Grouper(key='Date',freq='W')])['Total'].sum().reset_index()


# In[32]:


mean_total_sales = weekly_data.groupby('Date')['Total'].mean().reset_index()
mean_total_sales.rename(columns={'Total': 'Mean Total Sales'}, inplace = True)


# In[33]:


plt.figure(figsize = (20,10))
sns.lineplot(x='Date',y='Total',hue='City',data=weekly_data,marker='o',linewidth=3)
plt.title('Weekly Total Sales by City', fontsize = 20)
plt.xlabel('Date', fontsize = 16)
plt.ylabel('Total Sales', fontsize = 16)
plt.grid(True)
plt.legend(title = 'City', title_fontsize = '13', loc='upper left',fontsize='12')
plt.show()


# #### Comparison of Sales between Branches

# In[34]:


df['Month'] = df['Date'].dt.month


# In[35]:


grouped_data = df.groupby(['City','Month'])['Total'].sum().reset_index()


# In[36]:


plt.figure(figsize = (15,6))
sns.barplot(x='Month',y='Total',hue='City',data=grouped_data)
plt.title('Comparison of Sales income by Month and City')
plt.xlabel('Month')
plt.ylabel('Total')
plt.legend(title='City',loc='upper right')
plt.show()


# ### PREDICTION

# In[37]:


df = pd.read_csv('dataset.csv')


# In[40]:


df['Date'] = pd.to_datetime(df['Date'])

df['day_of_week'] = df['Date'].dt.dayofweek
df['day_of_month'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

df['Time'] = pd.to_datetime(df['Time'])

def map_time_interval(time):
    hour = time.hour
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'
    
df['time_interval'] = df['Time'].apply(map_time_interval)


# In[41]:


data_encoded = pd.get_dummies(df,columns=['City','Customer type','Gender','Product line','time_interval'],drop_first=True)


# In[42]:


data_encoded.drop(['Invoice ID','Date','Time','Tax 5%','gross margin percentage','cogs','year'],axis=1,inplace=True)


# In[43]:


categorical_columns = data_encoded.select_dtypes(include=['object']).columns


# In[46]:


categorical_columns


# In[48]:


data_encoded = pd.get_dummies(data_encoded,columns = categorical_columns,drop_first = True)


# In[49]:


X = data_encoded.drop('Rating',axis=1)
y = data_encoded['Rating']


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)


# In[51]:


scaler = StandardScaler()
X_train = scaler.fit_transform (X_train)
X_test = scaler.transform(X_test)


# In[52]:


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Neural Network": MLPRegressor(max_iter = 1000)
}

results = {}

for name, model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2
    print(f"{name}:{r2}")
    


# In[53]:


results_df = pd.DataFrame({"Model": list(results.keys()), "R2 Score": list(results.values())})
results_df = results_df.sort_values(by="R2 Score", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="R2 Score", y="Model", palette="viridis")
plt.title("Model R2 Score Comparison")
plt.xlabel("R2 Score")
plt.ylabel("Model")
plt.show()


# The results show that all the models have negative R2 scores, which indicates that they are worse than predicting the mean value of the target variable. This suggests that our current approach is not effective in predicting the 'Rating' variable.
# 
# There could be several reasons for this poor performance:
# The features in the dataset might not have a strong relationship with the 'Rating' variable. We might need to collect additional features or create better feature interactions to improve the model performance.
# 
# The default hyperparameters used for the models might not be optimal for this dataset. We could try performing hyperparameter tuning using methods like GridSearchCV or RandomizedSearchCV to find the best set of hyperparameters for each model.
# 
# The dataset is small (1,000 rows), the models might struggle to learn the underlying patterns. We could try collecting more data or using techniques like data augmentation to increase the size of the training dataset.
# 
# It is also possible that the 'Rating' variable is inherently difficult to predict based on the available features. In this case, we may need to reconsider the problem definition or the target variable itself.
