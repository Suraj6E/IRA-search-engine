import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pandas;

from Common import lemmatize_text




index_filename = 'search_indexs.json';
csv_filename = 'data.csv'

#read indverted Index file
def read_inverse_indexs():
    date = []
    with open(index_filename, 'r') as file:
        data = json.load(file)

    return data;


# task 1: make token words for query and filter data from inverse index
# task 2: among filter data get the rank using cosign similaritites


def filter_relevent_docs(query):
    inverse_index = read_inverse_indexs();
    query = lemmatize_text(query);

    #get filtered data
    filtered_data = [inverse_index.get(word) for word in query.split()]

    #get filtered data as document
    combined_array = []
    for item in filtered_data:
        if item is None: continue;
        combined_array = combined_array + item;

    combined_array = np.array(list(set(combined_array)))

    # # Extract the sets of values
    # value_sets = value_sets = [set(dictionary[key]) for dictionary in filtered_data for key in dictionary if dictionary[key] is not None]

    # # Find the common values among all sets
    # common_values = set.intersection(*value_sets)

    # data = {
    #     'OR': common_values,
    #     'AND': combined_array
    # }


    data = get_data_from_csv(combined_array);
    return data;

def get_data_from_csv(array):
    df = pandas.read_csv(csv_filename)
    
    index_array = np.array(array)
    index_list = index_array.tolist();

    selected_rows = df.iloc[index_list].copy()
    return selected_rows;


def get_relevent_score(query):
    df = filter_relevent_docs(query);

    data = df.to_dict('records')
    return data



#print(get_relevent_score("researching health"));



# # Document Collection
# documents = [
#     "I like to play soccer.",
#     "I enjoy watching movies.",
#     "I like playing soccer and watching movies."
# ]

# # Create the TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(documents)

# # User Query
# user_query = "play soccer"

# # Vectorize the query
# query_vector = vectorizer.transform([user_query])

# # Get relevant document indices from inverted index
# relevant_doc_ids = set()
# for term in user_query.split():
#     if term in inverted_index:
#         relevant_doc_ids.update(inverted_index[term])

# # Calculate cosine similarity between query and relevant documents
# cosine_similarities = cosine_similarity(query_vector, tfidf_matrix[relevant_doc_ids]).flatten()

# # Map sorted indices back to original document indices
# sorted_indices = np.argsort(cosine_similarities)[::-1]

# # Print the search results
# for index in sorted_indices:
#     original_index = relevant_doc_ids[index]
#     print(f"Document {original_index + 1}: {documents[original_index]} (Similarity: {cosine_similarities[index]})")