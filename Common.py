import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def lemmatize_text(input_text):
    input_text = input_text.lower();
    # Remove extra symbols from the input text
    input_text = re.sub(r'\W+', ' ', input_text);
    # Remove digits from the input text
    input_text = re.sub(r'\d+', '', input_text)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(input_text)
    output_text = []
    
    for token, tag in pos_tag(tokens):
        if token.lower() not in stop_words and token.isalpha():
            pos = get_wordnet_pos(tag)
            if pos:
                lemmatized_token = lemmatizer.lemmatize(token, pos=pos)
            else:
                lemmatized_token = lemmatizer.lemmatize(token)
            
            output_text.append(lemmatized_token)
    
    return " ".join(output_text);

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None