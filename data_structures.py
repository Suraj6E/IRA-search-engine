#create indexes 
import pandas as pandas;

import nltk
nltk.download('wordnet')
import re

from nltk.stem import WordNetLemmatizer



#should be called after new data is arrived
def get_df():
    return pandas.read_csv('data.csv');

def get_data_size():
    df = pandas.read_csv('data.csv');
    return df.size;


def create_index(df):
    if 'id' not in df.columns:
        df.insert(0, 'id', df.reset_index().index)
        df.to_csv('data.csv', index=False);
    return df;

def process_word(df):
    lemmatizer = WordNetLemmatizer()

    # Remove non-alphanumeric characters and make words lowercase for each words
    def lemmatize_word(word):
        word = re.sub(r'\W+', ' ', word)
        word = word.lower();
        return lemmatizer.lemmatize(word)

    df['title'] = df['title'].apply(lambda x: ' '.join([lemmatize_word(word) for word in x.split()]));

    return df;



df = create_index(get_df());
print(df['title'].head());

df = process_word(df);
# df.to_csv('data.csv', index=False)
print(df['title'].head());
