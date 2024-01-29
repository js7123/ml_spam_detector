from django.shortcuts import render
from django.http import JsonResponse
import os
import pickle
import urllib.parse

def index(request):
    return render(request, 'detector/index.html')

def predict(request):
    if request.method == 'POST':
        uri_encoded_message = request.POST.get('message')
        message = urllib.parse.unquote(uri_encoded_message, errors='strict')
        
        # Debugging statement
        print(f"Input POST request message: {message}")
        
        # Load the saved model and vectorizer
        model_path = os.path.join(os.path.dirname(__file__), 'spam_detector_model.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), 'count_vectorizer.pkl')
        with open(model_path, 'rb') as model_file:
            clf = pickle.load(model_file)
        with open(vectorizer_path, 'rb') as vectorizer_file:
            count_vectorizer = pickle.load(vectorizer_file)
            
        # Make a prediction
        message_vector = count_vectorizer.transform([message])
        prediction = clf.predict(message_vector)
        
        # Debugging statements
        # print(f"Input message: {message}")
        # features = list(count_vectorizer.vocabulary_.keys())
        # print(f"Vectorizer features: {features}")
        
        print(f"Message vector: {message_vector.toarray()}")
        print(f"Prediction: {prediction[0]}")
        
        return JsonResponse({'prediction': prediction[0]})
    else:
        return JsonResponse({'error': 'Invalid request method'})
