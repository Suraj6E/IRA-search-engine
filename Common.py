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

def lemmatize_sentance(sentance):
    sentance = sentance.lower();
    # Remove extra symbols from the sentance
    sentance = re.sub(r'\W+', ' ', sentance);
    # Remove digits from the sentance
    sentance = re.sub(r'\d+', '', sentance)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(sentance)
    lemmatized_tokens = []
    
    for token, tag in pos_tag(tokens):
        if token.lower() not in stop_words and token.isalpha():
            pos = get_wordnet_pos(tag)
            if pos:
                lemmatized_token = lemmatizer.lemmatize(token, pos=pos)
            else:
                lemmatized_token = lemmatizer.lemmatize(token)
            
            lemmatized_tokens.append(lemmatized_token)
    
    return lemmatized_tokens

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