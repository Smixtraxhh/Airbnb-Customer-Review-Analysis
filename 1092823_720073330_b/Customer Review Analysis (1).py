#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libarires
import pandas as pd
import numpy as np


# In[2]:


fd=pd.read_csv("D:\\reviews.csv")
listing_data = pd.read_excel("D:\\listing.xlsx")


# Pre processing and cleaning
# 

# In[3]:


fd.head(10)


# In[4]:


fd.info


# In[5]:


listing_data.info


# In[6]:


#checking null values
null_count = fd.isnull().sum()
print(null_count)



# In[7]:


#assuming dataframe as fd
#removing null values from rows
fd_without_null = fd.dropna()


# In[8]:


#removing null values from columns
fd_without_null = fd.dropna(axis = 1)


# In[9]:


# Remove rows only if all values in the row are null
fd_without_null = fd.dropna(how='all')


# In[10]:


# Remove rows if a specific column has null values
fd_without_null = fd.dropna(subset=['comments'])


# In[11]:


print(fd_without_null)


# In[12]:


# Assuming you have a DataFrame named 'df' without null values
# Check for null values in the DataFrame
null_counts = fd_without_null.isnull().sum()

# Print the null value counts
print(null_counts)


# In[13]:


# Importing the required libraries and methods
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime as dt

# Importing the dataset

reviews_filename = "D:\\reviews.csv"

data = pd.read_csv(reviews_filename, low_memory=False)


# Specifying data types for columns with mixed types
dtype_dict = {'column_name': str, 'other_column': int}  # Replace with actual column names and data types

# Loading the dataset

reviews = pd.read_csv(reviews_filename, names=['listing_id', 'comments'])


# Separating positive and negative reviews

# In[34]:


import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Assuming you have a DataFrame named 'df' with a column 'reviews' containing the customer reviews
# Read the customer reviews column into a list
reviews = fd['comments'].astype(str).tolist()

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

# Print the number of positive and negative reviews
print(f"Number of positive reviews: {len(positive_reviews)}")
print(f"Number of negative reviews: {len(negative_reviews)}")
# After sentiment classification
df_negative_reviews = pd.DataFrame({'negative_reviews': negative_reviews})


# Question 1 

# In[39]:


import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Sentiment analysis 
sia = SentimentIntensityAnalyzer()

reviews = fd['comments'].tolist()

positive_reviews = []
negative_reviews = []

for review in reviews:
   sentiment_scores = sia.polarity_scores(review)
   if sentiment_scores['compound'] >= 0:
       positive_reviews.append(review)
   else:
       negative_reviews.append(review)
       
# Create df_negative_reviews DataFrame
df_negative_reviews = pd.DataFrame({'negative_reviews': negative_reviews})

# Now access the column
negative_reviews = df_negative_reviews['negative_reviews'].tolist()

print(len(negative_reviews))


# In[15]:


nltk.download('stopwords')


# In[16]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Assuming you have a DataFrame named 'fd' with a column 'comments' containing the customer reviews
# Read the customer reviews column into a list
reviews = fd['comments'].astype(str).tolist()

# Initialize an empty Counter object
word_counts = Counter()

# Process the reviews in batches
batch_size = 1000  # Adjust the batch size as needed
num_reviews = len(reviews)
num_batches = (num_reviews // batch_size) + 1

# Add punctuation marks you want to exclude from the analysis to the set of stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '!'])

for i in range(num_batches):
    start_index = i * batch_size
    end_index = (i + 1) * batch_size

    # Combine the reviews in the batch into a single string
    batch_reviews = ' '.join(reviews[start_index:end_index])

    # Tokenize the text into individual words
    tokens = word_tokenize(batch_reviews)

    # Remove stopwords (common words like 'the', 'and', 'is') and punctuation marks
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]

    # Update the word counts
    word_counts.update(filtered_tokens)

# Get the most frequently mentioned topics (words)
most_common_topics = word_counts.most_common(10)  # Change the number to get more or fewer topics

# Print the most frequently mentioned topics
for topic, count in most_common_topics:
    print(f"Topic: {topic}, Count: {count}")


# Topic modelling on negative reviews 

# In[17]:


import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download the VADER lexicon and the stopwords (if not already downloaded)
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Assuming you have a DataFrame named 'df' with a column 'comments' containing the customer reviews
# Read the customer reviews column into a list
reviews = fd['comments'].astype(str).tolist()

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Create empty lists for positive and negative reviews
positive_reviews = []
negative_reviews = []

# Classify the reviews as positive or negative
for review in reviews:
    sentiment_scores = sia.polarity_scores(review)
    if sentiment_scores['compound'] >= 0:
        positive_reviews.append(review)
    else:
        negative_reviews.append(review)

# Function to check if a text is in English
def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False

# Filter out non-English negative reviews
english_negative_reviews = [review for review in negative_reviews if is_english(review)]

# Perform text preprocessing on the negative reviews
stop_words = set(stopwords.words('english'))
negative_reviews_cleaned = []
for review in english_negative_reviews:
    tokens = word_tokenize(review.lower())  # Convert to lowercase and tokenize
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]  # Remove non-alphabetic tokens and stopwords
    negative_reviews_cleaned.append(" ".join(filtered_tokens))

# Use CountVectorizer to convert text into a matrix of token counts
vectorizer = CountVectorizer(max_features=1000)  # You can adjust the number of features (topics) you want
X = vectorizer.fit_transform(negative_reviews_cleaned)

# Use LatentDirichletAllocation to find topics in the negative reviews
num_topics = 5  # You can adjust the number of topics you want
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Print the top words for each topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx + 1}:")
    top_words_idx = topic.argsort()[:-11:-1]  # Get the indices of the top 10 words for the topic
    top_words = [feature_names[i] for i in top_words_idx]
    print(", ".join(top_words))


# In[40]:


import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect

# Read the negative reviews column into a list
negative_reviews = df_negative_reviews['negative_reviews'].astype(str).tolist()

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Analyze the negative reviews
for review in negative_reviews:
    # Detect the language of the review
    lang = detect(review)
    
    # Process only the English reviews
    if lang == 'en':
        sentiment_scores = sia.polarity_scores(review)
        
        # Access the sentiment scores
        compound_score = sentiment_scores['compound']
        negative_score = sentiment_scores['neg']
        neutral_score = sentiment_scores['neu']
        positive_score = sentiment_scores['pos']
        
        # Do further analysis or processing with the sentiment scores
        # For example, you can calculate the average sentiment scores, classify the intensity of negativity, etc.
        
        # Print the sentiment scores of each English negative review
        print(f"Review: {review}")
        print(f"Compound Score: {compound_score}")
        print(f"Negative Score: {negative_score}")
        print(f"Neutral Score: {neutral_score}")
        print(f"Positive Score: {positive_score}")
        print("-----------------------------------")


# Questions 2

# In[22]:


from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

fd['comments'] = fd['comments'].astype(str)  # Convert 'comments' column to string type

# Perform sentiment analysis on the comments
sentiments = fd['comments'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Add the sentiment scores to the DataFrame
fd['sentiment'] = sentiments


aspects = ['amenities', 'cleanliness', 'location', 'service']

aspect_reviews = {aspect: [] for aspect in aspects}

for index, row in fd.iterrows():
    for aspect in aspects:
        if aspect in row['comments'].lower():
            aspect_reviews[aspect].append(row['comments'])
correlation_results = {}



# Calculate the correlation between aspect sentiments and overall sentiments
for aspect, reviews in aspect_reviews.items():
    aspect_sentiments = fd.loc[fd['comments'].isin(reviews), 'sentiment']
    overall_sentiments = fd['sentiment']
    correlation = aspect_sentiments.corr(overall_sentiments)  # Calculate the correlation
    correlation_results[aspect] = correlation

# Print the correlation results for each aspect
for aspect, correlation in correlation_results.items():
    print(f"Correlation between {aspect} and overall sentiment: {correlation}")




# In[23]:


from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

fd['comments'] = fd['comments'].astype(str)  # Convert 'comments' column to string type

# Perform sentiment analysis on the comments
sentiments = fd['comments'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Add the sentiment scores to the DataFrame
fd['sentiment'] = sentiments

# Filter out only the negative sentiments
negative_sentiments = fd[fd['sentiment'] < 0]

# List of aspects (features)
aspects = ['amenities', 'cleanliness', 'location', 'service', 'value_for_money', 'communication', 'check-in_process']

# Create a dictionary to store reviews for each aspect
aspect_reviews = {aspect: [] for aspect in aspects}

# Separate the reviews for each aspect
for index, row in negative_sentiments.iterrows():
    for aspect in aspects:
        if aspect in row['comments'].lower():
            aspect_reviews[aspect].append(row['comments'])

correlation_results = {}

# Calculate the correlation between aspect sentiments and negative sentiments
for aspect, reviews in aspect_reviews.items():
    aspect_sentiments = negative_sentiments[negative_sentiments['comments'].isin(reviews)]['sentiment']
    overall_sentiments = negative_sentiments['sentiment']
    correlation = aspect_sentiments.corr(overall_sentiments)  # Calculate the correlation
    correlation_results[aspect] = correlation

# Print the correlation results for each aspect
for aspect, correlation in correlation_results.items():
    print(f"Correlation between {aspect} and negative sentiment: {correlation}")


# In[25]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Read the datasets
fd = pd.read_csv("D:\\reviews.csv")
listing_data = pd.read_excel("D:\\listing.xlsx")

# Convert 'comments' column to string type
fd['comments'] = fd['comments'].astype(str)

# Perform sentiment analysis on the comments
sentiments = fd['comments'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Add the sentiment scores to the DataFrame
fd['sentiment'] = sentiments

# Merge the datasets based on 'id' from 'listing_data' and 'listing_id' from 'fd'
merged_data = pd.merge(fd, listing_data, left_on='listing_id', right_on='id')


# In[26]:


# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(merged_data['review_scores_rating'], merged_data['sentiment'], alpha=0.5)
plt.xlabel('Review Scores Rating')
plt.ylabel('Sentiment Score')
plt.title('Scatter Plot: Sentiment Scores vs Review Scores Rating')
plt.show()


# In[27]:


# Grouped bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='room_type', y='sentiment', data=merged_data)
plt.xlabel('Room Type')
plt.ylabel('Sentiment Score')
plt.title('Grouped Bar Plot: Sentiment Scores by Room Type')
plt.show()


# In[28]:


# Create a crosstab to get the count of sentiment categories by neighborhood
sentiment_by_neighborhood = pd.crosstab(merged_data['neighbourhood'], merged_data['sentiment'])

# Stacked bar plot
plt.figure(figsize=(12, 6))
sentiment_by_neighborhood.plot(kind='bar', stacked=True, cmap='coolwarm')
plt.xlabel('Neighborhood')
plt.ylabel('Count')
plt.title('Stacked Bar Plot: Sentiment Categories by Neighborhood')
plt.legend(title='Sentiment', loc='upper right', labels=['Negative', 'Neutral', 'Positive'])
plt.show(block=True)


# In[29]:


# Select numerical features for correlation heatmap
numerical_features = merged_data.select_dtypes(include='number')

# Calculate correlation matrix
correlation_matrix = numerical_features.corr()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the 'reviews.csv' file and replace NaN values in 'comments' with an empty string
fd = pd.read_csv("D:\\reviews.csv", dtype={'comments': str})
fd['comments'].fillna('', inplace=True)

# Combine all comments into a single string
all_comments = ' '.join(fd['comments'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Customer Reviews')
plt.show()


# In[32]:


# Assuming you have 'latitude' and 'longitude' columns in the 'listing_data' DataFrame

plt.figure(figsize=(10, 8))
sns.scatterplot(x='longitude', y='latitude', data=listing_data, hue='property_type', palette='Set1', s=50)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Distribution of Listings')
plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

