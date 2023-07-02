import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np





index_filename = 'search_indexs.json';

#read indverted Index file
def read_inverse_indexs():
    date = []
    with open(index_filename, 'r') as file:
        data = json.load(file)

    return data;


# task 1: make token words for query and filter data from inverse index
# task 2: among filter data get the rank using cosign similaritites


def filter_relevent_docs():
    return data;

def read_data_using_index(index): 
    return data;


#inverted_index = read_indexs();


# Document Collection
documents = [
    "I like to play soccer.",
    "I enjoy watching movies.",
    "I like playing soccer and watching movies."
]

# Inverted Index
inverted_index = {
    'like': [0, 2],
    'play': [0, 2],
    'soccer': [0, 2],
    'enjoy': [1],
    'watching': [1, 2],
    'movies': [1, 2]
}

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# User Query
user_query = "play soccer"

# Vectorize the query
query_vector = vectorizer.transform([user_query])

# Get relevant document indices from inverted index
relevant_doc_ids = set()
for term in user_query.split():
    if term in inverted_index:
        relevant_doc_ids.update(inverted_index[term])

# Calculate cosine similarity between query and relevant documents
cosine_similarities = cosine_similarity(query_vector, tfidf_matrix[relevant_doc_ids]).flatten()

# Map sorted indices back to original document indices
sorted_indices = np.argsort(cosine_similarities)[::-1]

# Print the search results
for index in sorted_indices:
    original_index = relevant_doc_ids[index]
    print(f"Document {original_index + 1}: {documents[original_index]} (Similarity: {cosine_similarities[index]})")