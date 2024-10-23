import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the dataset
#df = pd.read_csv('books.csv')  # You'll need to download this dataset
df = pd.read_csv('/data/books.csv')

# Preprocess the data
df['features'] = df['authors'] + ' ' + df['genres']
df['features'] = df['features'].fillna('')

# Create TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get book recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar books
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices]

# Initialize sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to perform sentiment analysis
def analyze_sentiment(text):
    return sia.polarity_scores(text)['compound']

# Main function
def main():
    book_title = input("Enter a book title: ")
    recommendations = get_recommendations(book_title)
    
    print(f"\nTop 5 Recommendations for '{book_title}':")
    for i, title in enumerate(recommendations[:5], 1):
        # In a real scenario, you'd fetch actual reviews here
        mock_review = f"This is a mock review for {title}"
        sentiment = analyze_sentiment(mock_review)
        print(f"{i}. {title} (Sentiment Score: {sentiment:.2f})")

if __name__ == "__main__":
    main()
