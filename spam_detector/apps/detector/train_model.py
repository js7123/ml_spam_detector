import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
def train_model():
    #Load dataset
    data = pd.read_csv('data/SMSSpamCollection', sep='\t', names=['label', 'message'])
    
    # Feature extraction

    count_vectorizer = CountVectorizer(stop_words='english')
    X = count_vectorizer.fit_transform(data['message'])
    y = data['label']

    # Split dataset into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    # Train the model

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Evaluate the model

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model and vectorizer

    with open('spam_detector_model.pkl', 'wb') as model_file:
        pickle.dump(clf, model_file)
    with open('count_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(count_vectorizer, vectorizer_file)

if __name__ == "__main__":
    train_model()