import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
from dash import Dash, dcc, html
import plotly.express as px
from pandas.core.common import flatten
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.collocations import *

# Load the csv file
reviews_df = pd.read_csv('B01N4B5MRD.csv')

# Drop rows containing NaN values in the 'Body' column
reviews_df.dropna(subset=['Body'], inplace=True)

# Convert the 'Body' column to string format
reviews_df['Body'] = reviews_df['Body'].astype(str)

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Add a new column to the dataframe with the sentiment scores
reviews_df['Sentiment'] = reviews_df['Body'].apply(
    lambda x: sid.polarity_scores(str(x)) if type(x) == float else sid.polarity_scores(x))

# Add new columns for each sentiment score
reviews_df['Positive Score'] = reviews_df['Sentiment'].apply(lambda x: x['pos'])
reviews_df['Negative Score'] = reviews_df['Sentiment'].apply(lambda x: x['neg'])
reviews_df['Neutral Score'] = reviews_df['Sentiment'].apply(lambda x: x['neu'])
reviews_df['Compound Score'] = reviews_df['Sentiment'].apply(lambda x: x['compound'])

# Calculate the percentage of positive, negative, and neutral reviews
positive_reviews = len(reviews_df[reviews_df['Compound Score'] > 0])
negative_reviews = len(reviews_df[reviews_df['Compound Score'] < 0])
neutral_reviews = len(reviews_df[reviews_df['Compound Score'] == 0])

total_reviews = positive_reviews + negative_reviews + neutral_reviews

positive_percentage = (positive_reviews / total_reviews) * 100
negative_percentage = (negative_reviews / total_reviews) * 100
neutral_percentage = (neutral_reviews / total_reviews) * 100

# Extract top feature requests
nouns = []
for review in reviews_df['Body']:
    if type(review) == str:  # Check if the value is a string
        tokens = word_tokenize(review)
        tagged = pos_tag(tokens)
        for word, tag in tagged:
            if tag.startswith('NN'):
                nouns.append(word)

features = []
for chunk in ne_chunk(pos_tag(nouns)):
    if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
        features.append(' '.join(c[0] for c in chunk))

feature_counts = pd.Series(features).value_counts()
top_feature_requests = feature_counts[:6].index.tolist()

# Extract top feature complaints
complaints = []
for review in reviews_df['Body']:
    if 'not' in str(review):
        tokens = word_tokenize(review)
        tagged = pos_tag(tokens)
        for i, (word, tag) in enumerate(tagged):
            if word == 'not':
                for j in range(i + 1, len(tagged)):
                    if tagged[j][1].startswith('NN'):
                        complaints.append(tagged[j][0])
                        break

complaint_counts = pd.Series(complaints).value_counts()
top_feature_complaints = complaint_counts[:5].index.tolist()

# Create a dictionary with the results
results = {
    'positive_percentage': positive_percentage,
    'negative_percentage': negative_percentage,
    'neutral_percentage': neutral_percentage,
    'top_feature_requests': top_feature_requests,
    'top_feature_complaints': top_feature_complaints
}

# Extract the text column from the DataFrame
text = ' '.join(reviews_df['Body'].tolist())

# Tokenize the text
sentences = sent_tokenize(text)
words = word_tokenize(text)

stop_words = set(stopwords.words("english"))

filtered_list = [
    word for word in words if word.casefold() not in stop_words
]

stemmer = PorterStemmer()

# Join the list of words into a single string
filtered_text = ' '.join(filtered_list)

words = word_tokenize(filtered_text)

stemmed_words = [stemmer.stem(word) for word in words]

word_word = nltk.pos_tag(words)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(word_word)
nltk.download("maxent_ne_chunker")
nltk.download("words")
tree = nltk.ne_chunk(word_word, binary=True)

frequency_distribution = FreqDist(words)
print(frequency_distribution)
frequency_distribution.most_common(20)
frequency_distribution.plot(20, cumulative=True)
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
new_text = nltk.Text(words)
new_text.collocations()


with open('results.txt', 'w') as file:
    file.write(f"Positive Percentage: {results['positive_percentage']:.2f}%\n")
    file.write(f"Negative Percentage: {results['negative_percentage']:.2f}%\n")
    file.write(f"Neutral Percentage: {results['neutral_percentage']:.2f}%\n\n")

    file.write("Top Feature Requests:\n")
    for feature in results['top_feature_requests']:
        file.write(f"- {feature}\n")
    file.write("\n")

    file.write("Top Feature Complaints:\n")
    for complaint in results['top_feature_complaints']:
        file.write(f"- {complaint}\n")
    file.write("\n")

    file.write(f"Reviews Summary: {new_text}\n")


# Create pie chart for sentiment analysis results
labels = ['Positive', 'Negative', 'Neutral']
values = [positive_percentage, negative_percentage, neutral_percentage]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

# Update the layout
fig.update_layout(
    title='Sentiment Analysis Results',
    template='seaborn',  # Use the seaborn template for a better-looking chart
    annotations=[{
        'text': f"",
        'x': 0.5,
        'y': 0.5,
        'font_size': 20,
        'showarrow': False}]
)

# Save the chart as an HTML file
fig.write_html('sentiment_analysis_results.html')


