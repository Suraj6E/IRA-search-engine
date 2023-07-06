from Common import lemmatize_text
import pandas as pandas;


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt



filename = "articles.csv"
chart_path = 'static/classification.png'


def read_csv():
    return pandas.read_csv(filename)

def preprocess(df):
    # Remove special characters
    df["article_temp"] = df["article"].replace("\n", " ");
    df["article_temp"] = df["article_temp"].replace("\r", " ");
    df['article_temp'] = df['article_temp'].apply(lambda x: lemmatize_text(x));
    return df;

def naive_bayes_classification(query_text):
    
    df = read_csv();
    df = preprocess(df);

    categories = df['category']
    articles = df['article_temp']

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(articles)

    # Train a Naïve Bayes classifier
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X, categories)

    # Transform the query text into TF-IDF representation
    query_text_tfidf = vectorizer.transform([query_text])

    # Predict the label probabilities for the new document
    predicted_probabilities = naive_bayes.predict_proba(query_text_tfidf)[0]

    # Get the predicted label scores
    categories_scores = dict(zip(naive_bayes.classes_, predicted_probabilities))

    # Predict the label for the new document
    predicted_category = naive_bayes.predict(query_text_tfidf)

    #createn a pie chart for UI
    create_save_chart(naive_bayes, predicted_probabilities);

    return {
        "predicted_category": predicted_category[0],
        "img": chart_path,
        "predicted_probabilities": predicted_probabilities,
    }


def create_save_chart(naive_bayes, predicted_probabilities):
    # Prepare data for pie chart
    labels = naive_bayes.classes_
    sizes = predicted_probabilities
    
    # Plotting the pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')  # Equal aspect ratio ensures a circular pie chart
    plt.title('Predicted Categories Probabilities')
    #plt.show()

    # Save the pie chart to a file
    plt.savefig(chart_path)

#naive_bayes_classification("More work is needed to understand why the rise is happening, they say. Some of the rise could be attributed to catch-up - from backlogs and delays when health services were shut - but does not explain all of the newly diagnosed cases, say scientists.");