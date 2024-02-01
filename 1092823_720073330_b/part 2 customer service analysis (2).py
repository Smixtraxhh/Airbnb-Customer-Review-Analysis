#!/usr/bin/env python
# coding: utf-8

# Question 3

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[3]:


# Assuming you have two files, one CSV and one Excel file
reviews_data = pd.read_csv("D:\\reviews.csv", encoding='utf-8')
listing_data = pd.read_excel("D:\\listing.xlsx")



# In[4]:


# Assuming you have two DataFrames named 'listing_data' and 'reviews_data'
# Merge the listing data and review data based on the appropriate column name
unified_data = pd.merge(listing_data, reviews_data, left_on='id', right_on='listing_id')

# Check the unified dataset
print(unified_data.head())


# In[5]:


# Select the relevant columns for analysis
selected_columns = ['listing_id', 'comments', 'review_scores_rating']

# Create a new DataFrame with the selected columns
selected_data = unified_data[selected_columns]

# Check the selected data
print(selected_data.head())


# In[6]:


import nltk
nltk.download('vader_lexicon')


# In[7]:


print("Listing Data Columns:", listing_data.columns)
print("Reviews Data Columns:", reviews_data.columns)


# In[8]:


# Rename the 'id' column in listing_data to 'listing_id'
listing_data.rename(columns={'id': 'listing_id'}, inplace=True)


# In[9]:


# Convert the 'listing_id' column in both DataFrames to the same data type (e.g., int)
listing_data['listing_id'] = listing_data['listing_id'].astype(int)
reviews_data['listing_id'] = reviews_data['listing_id'].astype(int)


# In[10]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Assuming you have the NLTK library and VADER lexicon installed
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Merge the listing data and review data based on the 'listing_id' column
unified_data = pd.merge(listing_data, reviews_data, on='listing_id')

# Convert non-string values in the 'comments' column to empty strings
unified_data['comments'] = unified_data['comments'].astype(str)

# Perform sentiment analysis on the comments
sentiments = unified_data['comments'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Add the sentiment scores to the DataFrame
unified_data['sentiment'] = sentiments

# Print the first few rows of the DataFrame with sentiment scores
print(unified_data[['comments', 'sentiment']].head(15))

# Now, let's proceed with the merge
# Merge the sentiment scores with the 'listing_data' DataFrame based on the 'listing_id' column
merged_data = pd.merge(listing_data, unified_data[['listing_id', 'sentiment']], on='listing_id', how='left')

# Check the merged DataFrame
print(merged_data.head())


# In[11]:


import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Assuming you have a DataFrame named 'df' with a column 'comments' containing the customer reviews
# Read the customer reviews column into a list
reviews = reviews_data['comments'].astype(str).tolist()

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Create empty lists for positive and negative reviews
positive_reviews = []
negative_reviews = []

# Classify the reviews as positive or negative
for review in reviews:
    sentiment_scores = sia.polarity_scores(review)
    
    # Assign the review to positive or negative based on the compound score
    if sentiment_scores['compound'] >= 0:
        positive_reviews.append(review)
    else:
        negative_reviews.append(review)

# Store the negative reviews in a separate DataFrame or list
df_negative_reviews = pd.DataFrame({'negative_reviews': negative_reviews})  # If using a DataFrame
# negative_reviews_list = negative_reviews  # If using a list

# Print the number of positive and negative reviews
print(f"Number of positive reviews: {len(positive_reviews)}")
print(f"Number of negative reviews: {len(negative_reviews)}")


# In[12]:


import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Assuming you have the NLTK library and VADER lexicon installed
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Assuming you have stored negative reviews in a DataFrame named 'df_negative_reviews'
negative_sentiments = df_negative_reviews['negative_reviews'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Add the sentiment scores to the DataFrame
df_negative_reviews['sentiment'] = negative_sentiments

# Print the first few rows of the DataFrame with sentiment scores for negative reviews
print(df_negative_reviews[['negative_reviews', 'sentiment']].head(15))


# In[13]:


# Assuming you have stored negative reviews in a DataFrame named 'df_negative_reviews'
negative_sentiments = df_negative_reviews['negative_reviews'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Print the first few sentiment scores for negative reviews
print(negative_sentiments.head(15))


# In[14]:


listing_data['listing_id'] = listing_data['listing_id'].astype('int32')


# In[15]:


listing_data['listing_id'] = listing_data['listing_id'].astype('int32')

# Merge the sentiment scores with the 'listing_data' DataFrame based on the 'listing_id' column
merged_data = pd.merge(listing_data, unified_data[['listing_id', 'sentiment']], on='listing_id', how='left')

# Check the merged DataFrame
print(merged_data.head())


# In[16]:


# Print the first few rows of the DataFrame with sentiment scores and review ratings
print(merged_data[['listing_id', 'sentiment', 'review_scores_rating']].head())


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have already merged the sentiment scores with the 'listing_data' DataFrame
# Let's calculate the average sentiment score and average review rating for each listing
grouped_data = merged_data.groupby('listing_id').agg({
    'sentiment': 'mean',
    'review_scores_rating': 'mean'
}).reset_index()

# Create a scatter plot to visualize the relationship between sentiment scores and review ratings
plt.figure(figsize=(8, 6))
sns.scatterplot(data=grouped_data, x='sentiment', y='review_scores_rating')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Average Review Rating')
plt.title('Sentiment Score vs. Review Rating')
plt.show()

# Plot the distribution of sentiment scores
plt.figure(figsize=(8, 6))
sns.histplot(data=grouped_data, x='sentiment', bins=20, kde=True)
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Scores')
plt.show()

# Plot the distribution of review ratings
plt.figure(figsize=(8, 6))
sns.histplot(data=grouped_data, x='review_scores_rating', bins=20, kde=True)
plt.xlabel('Review Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Review Ratings')
plt.show()


# In[18]:


import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string

# Assuming you have already merged the sentiment scores with the 'listing_data' DataFrame
# Let's create a new DataFrame to store the additional features
feature_engineered_data = merged_data.copy()

# Initialize the VADER sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Preprocess the reviews
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

# Assuming 'unified_data' contains the 'comments' column
unified_data['comments'] = unified_data['comments'].astype(str)
feature_engineered_data['comments'] = unified_data['comments'].apply(preprocess_text)

# Define lists of positive and negative words (you can add more words to these lists)
positive_words = ['good', 'excellent', 'amazing', 'great', 'wonderful', 'happy']
negative_words = ['bad', 'poor', 'terrible', 'awful', 'horrible', 'disappointing']

# Assuming 'positive_words' and 'negative_words' are the lists of positive and negative words
# that you have defined

# Count positive and negative words in the reviews
feature_engineered_data['comments'] = feature_engineered_data['comments'].fillna('')  # Handle missing values
feature_engineered_data['positive_word_count'] = feature_engineered_data['comments'].apply(lambda x: sum(1 for word in x.split() if word in positive_words))
feature_engineered_data['negative_word_count'] = feature_engineered_data['comments'].apply(lambda x: sum(1 for word in x.split() if word in negative_words))


# In[19]:


# Print the first few rows of the DataFrame with additional features
print(feature_engineered_data.head(20000))


# In[20]:


print(feature_engineered_data.columns)


# In[21]:


column_names = feature_engineered_data.columns.to_list()
print(column_names)


# In[22]:


# Prepare X and y after dropping rows
X = feature_engineered_data[['sentiment', 'positive_word_count', 'negative_word_count', 'review_scores_rating']]
y = feature_engineered_data['review_scores_rating']

print("Shape of X:", X.shape)
print("Number of null values in X:")
print(X.isnull().sum())
print("\nShape of y:", y.shape)
print("Number of null values in y:", y.isnull().sum())


# In[23]:


from sklearn.impute import SimpleImputer

# Handle missing values in y
imputer = SimpleImputer(strategy='mean')
y = imputer.fit_transform(y.values.reshape(-1, 1))  # Reshape y to a 2D array

# Prepare the dataset (ensure feature_engineered_data contains all necessary features)
X = feature_engineered_data[['sentiment', 'positive_word_count', 'negative_word_count', 'review_scores_rating']]


# In[24]:


print(X.columns)
print(X.columns.tolist())


# In[25]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

# Assuming you have already prepared the dataset and imputed missing values
# Assuming 'X' is your DataFrame containing the features, and 'y' is the target variable

# Encode the 'sentiment' column using one-hot encoding as a sparse matrix
sentiment_encoded = pd.get_dummies(X['sentiment'], sparse=True)
sentiment_columns = [f"sentiment_{category}" for category in sentiment_encoded.columns]

# Drop the original 'sentiment' column and concatenate one-hot encoded columns
X.drop(columns=['sentiment'], inplace=True)
X_sparse = hstack([csr_matrix(X.values), sentiment_encoded])

# Assuming you have prepared the target variable 'y' accordingly

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train_sparse, X_test_sparse, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2, random_state=42)

# Convert X_train_sparse to a dense numpy array
X_train = X_train_sparse.toarray()

# Create and train the model (HistGradientBoostingRegressor)
model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Convert X_test_sparse to a dense numpy array
X_test = X_test_sparse.toarray()

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate and print the mean squared error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


# In[26]:


pip install folium


# In[28]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset into the 'listing_data' DataFrame
# listing_data = pd.read_csv('your_data.csv')

# Calculate correlation matrix
correlation_matrix = listing_data.corr()

# Set a threshold for correlations to display
correlation_threshold = 0.5

# Create a correlation matrix with only values above the threshold
correlation_matrix_filtered = correlation_matrix.iloc[:50, :50][correlation_matrix.abs() > correlation_threshold]

# Increase the figure size
plt.figure(figsize=(10, 8))

# Plot the heatmap
sns.heatmap(correlation_matrix_filtered, annot=True, cmap='coolwarm', xticklabels=correlation_matrix_filtered.columns, yticklabels=correlation_matrix_filtered.columns, annot_kws={"rotation": 45})

plt.show()

# Alternatively, you can use the clustered heatmap approach
sns.clustermap(correlation_matrix_filtered, annot=True, cmap='coolwarm', figsize=(10, 8))

plt.show()

# If you want to use a different color palette
sns.heatmap(correlation_matrix_filtered, annot=True, cmap='RdBu_r', center=0)

plt.show()


# In[ ]:


print(listing_data['price'].head())
print(listing_data['price'].dtype)


# In[34]:


import matplotlib.pyplot as plt

# Bar chart of % positive sentiment by aspect
aspects = ['location', 'cleanliness', 'amenities', 'noise', 'parking', 'communication']
pos_percentages = [81, 75, 70, 30, 25, 20] 




# Horizontal bar chart of top negative aspects
neg_aspects = merged_data[merged_data['sentiment'] < 0]
counts = neg_aspects['sentiment'].value_counts()
counts.head(10).plot(kind='barh')
plt.title("Most Frequent Negative Aspects")


# In[36]:


import matplotlib.pyplot as plt

aspects = ['location', 'cleanliness', 'amenities', 'noise', 'parking', 'communication']
pos_percentages = [81, 75, 70, 30, 25, 20] 

plt.figure()
plt.bar(aspects, pos_percentages)
plt.title("Percentage of Positive Sentiment by Aspect")
plt.show()


# In[51]:


# Frequency table of total reviews by aspect
aspects = ['location', 'cleanliness', 'amenities', 'noise', 'parking', 'communication']
aspect_counts = merged_data['sentiment'].value_counts()
print(aspect_counts)

# Pivot table of mean sentiment by aspect
# Replace 'aspect' with the actual column name representing the aspect you want to analyze
aspect_column = 'reviews_per_month', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness' 
aspect_means = merged_data.pivot_table(values='sentiment', index=aspect_column) 
print(aspect_means)


# In[67]:


# Calculate the percentage of positive sentiment for each aspect
pos_percentages = []
for aspect in aspects:
    matching_rows = unified_data[unified_data['comments'].str.contains(aspect, case=False)]
    total_reviews = len(matching_rows)
    positive_reviews = len(matching_rows[matching_rows['sentiment'] > 0])
    
    print(f"Aspect: {aspect}, Total Reviews: {total_reviews}, Positive Reviews: {positive_reviews}")
    
    if total_reviews == 0:
        pos_percentage = 0
    else:
        pos_percentage = (positive_reviews / total_reviews) * 100
    
    pos_percentages.append(pos_percentage)

print(pos_percentages)


# In[69]:


# Bar chart of % positive sentiment by aspect
aspects = ['location', 'cleanliness', 'amenities', 'noise', 'parking', 'communication']
pos_percentages = [98.63, 97.40, 99.20, 95.36, 98.35, 97.16]

plt.figure()
plt.bar(aspects, pos_percentages)
plt.title("Percentage of Positive Sentiment by Aspect")


# In[68]:


import pandas as pd

aspects = ['location', 'cleanliness', 'amenities', 'noise', 'parking', 'communication']
total_reviews = [5775, 77, 377, 474, 243, 704]
positive_reviews = [5696, 75, 374, 452, 239, 684]
pos_percentages = [98.63, 97.40, 99.20, 95.36, 98.35, 97.16]

data = {
    'Aspect': aspects,
    'Total Reviews': total_reviews,
    'Positive Reviews': positive_reviews,
    'Positive Sentiment (%)': pos_percentages
}

df = pd.DataFrame(data)
print(df)


# In[ ]:




