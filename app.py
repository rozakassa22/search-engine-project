from flask import Flask, render_template, request
import os
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Directory containing the .txt files
directory = "documents"

docs = []
filenames = []

# Load documents and filenames
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filenames.append(filename)
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            docs.append(content)

# Preprocess the documents
english_stopset = set(stopwords.words('english')).union(
    {"things", "that's", "something", "take", "don't", "may", "want", "you're",
     "set", "might", "says", "including", "lot", "much", "said", "know",
     "good", "step", "often", "going", "thing", "things", "think",
     "back", "actually", "better", "look", "find", "right", "example",
     "verb", "verbs"})

stemmer = PorterStemmer()

documents_clean = []

for doc in docs:
    # Remove non-ASCII characters
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', doc)
    # Remove mentions
    document_test = re.sub(r'@\w+', '', document_test)
    # Convert to lowercase
    document_test = document_test.lower()
    # Remove punctuation
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    # Remove numbers
    document_test = re.sub(r'\d+', '', document_test)
    # Tokenize
    tokens = nltk.word_tokenize(document_test)
    # Remove stopwords and stem each token
    tokens = [stemmer.stem(token) for token in tokens if token not in english_stopset]
    # Rejoin tokens
    document_clean = ' '.join(tokens)
    documents_clean.append(document_clean)

# Build the TF-IDF vectorizer
vectorizer = TfidfVectorizer(analyzer='word',
                             ngram_range=(1, 2),
                             min_df=0.002,
                             max_df=0.99,
                             max_features=10000,
                             lowercase=True,
                             stop_words=list(english_stopset))  # Convert set to list

X = vectorizer.fit_transform(documents_clean)

@app.route('/')
def index():
    return render_template('index.html')

def get_snippet(doc_text, query_tokens, snippet_length=30):
    # Split the document into sentences
    sentences = nltk.sent_tokenize(doc_text)
    # Find sentences that contain the query terms
    for sentence in sentences:
        sentence_tokens = nltk.word_tokenize(sentence.lower())
        if any(token in sentence_tokens for token in query_tokens):
            # Return the sentence as the snippet
            return sentence
    # If no sentence contains the query, return the first snippet_length words
    words = nltk.word_tokenize(doc_text)
    snippet = ' '.join(words[:snippet_length])
    return snippet

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    # Process the query
    query_processed = re.sub(r'[^\x00-\x7F]+', ' ', query)
    query_processed = re.sub(r'@\w+', '', query_processed)
    query_processed = query_processed.lower()
    query_processed = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', query_processed)
    query_processed = re.sub(r'\d+', '', query_processed)
    # Tokenize and stem each token
    query_tokens = nltk.word_tokenize(query_processed)
    query_tokens = [stemmer.stem(token) for token in query_tokens if token not in english_stopset]
    query_clean = ' '.join(query_tokens)
    # Vectorize the query
    query_vec = vectorizer.transform([query_clean])
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarities = cosine_similarity(query_vec, X).flatten()
    # Get the top matching documents
    top_n = 10
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    matched_documents = []
    for index in related_docs_indices:
        if cosine_similarities[index] > 0:
            # Get snippet from the document
            snippet = get_snippet(docs[index], query_tokens)
            matched_documents.append({'filename': filenames[index], 'snippet': snippet, 'similarity': cosine_similarities[index]})
    # Number of documents that match the query
    num_matching_documents = len(matched_documents)
    # Render the results page
    return render_template('results.html', query=query, matched_documents=matched_documents, num_matching_documents=num_matching_documents)

if __name__ == '__main__':
    app.run(debug=True)
