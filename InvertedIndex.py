#create indexes 
import pandas as pandas;
import json
from collections import defaultdict
from Common import lemmatize_text


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
   
    df['title'] = df['title'].apply(lambda x: lemmatize_text(x));
    df['all_authors'] = df['all_authors'].apply(lambda x: lemmatize_text(x));

    print(df.head())
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
