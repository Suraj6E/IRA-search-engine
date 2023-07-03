#create indexes 
import pandas as pandas;
import re

import json
from collections import defaultdict

#lemmatizer's import
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
from nltk.corpus import stopwords



index_filename = 'search_indexs.json';
csv_filename = 'data.csv'

#should be called after new data is arrived
def get_df():
    return pandas.read_csv(csv_filename);

def get_data_size():
    df = pandas.read_csv(csv_filename);
    return df.size;


def add_index_col(df):
    if 'id' not in df.columns:
        df.insert(0, 'id', df.reset_index().index)
        df.to_csv(csv_filename, index=False);
    return df;

def process_word(df):
    lemmatizer = WordNetLemmatizer()

    # Remove non-alphanumeric characters and make words lowercase for each words
    def lemmatize_word(word):

        #collection of stopwords
        stop_words = set(stopwords.words('english'))
        
        word = word.lower();

        #filter stop words
        if(word in stop_words): return '';

        word = re.sub(r'\W+', ' ', word)
        return lemmatizer.lemmatize(word)


    df['title'] = df['title'].apply(lambda x: ' '.join([lemmatize_word(word) for word in x.split()]));
    df['all_authors'] = df['all_authors'].apply(lambda x: ' '.join([lemmatize_word(word) for word in x.split()]));

    return df;

def create_indexes(df):
    word_dict = defaultdict(list);
    for index, row in df.iterrows():
        title = row['title'];
        authors = row['all_authors'];
        words = title.split() + authors.split();
        for word in words:
            word_dict[word].append(index)
    return word_dict;


#utilized everything on this file
def create_save_indexes():
    try: 
        df = add_index_col(get_df());
        print("Processing words.");
        df = process_word(df);
        
        print("Creating Indexes.");
        indexes = create_indexes(df);

        print("Indexing finished successfully.");
        #save indexes
        with open(index_filename, 'w') as new_f:
            json.dump(indexes, new_f, sort_keys=True, indent=4)
        
        return True;
    except Exception as e:
        print(f"An error occurred on indexing: {str(e)}")
        return False;


#called from subproess
create_save_indexes();

#single_value = df.loc[0:1].copy();
