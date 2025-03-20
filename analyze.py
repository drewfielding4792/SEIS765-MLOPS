from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path

sentiment_pipeline = pipeline("sentiment-analysis")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def load_classes():
    with open('email_class.json', 'r') as file:
        data = json.load(file)
    return data['classes']


def get_sentiment(text):
    response = sentiment_pipeline(text)
    return response

def compute_embeddings(embeddings = load_classes()):
    embeddings = model.encode(embeddings)
    return zip(load_classes(), embeddings)

def classify_email(text):
    # Encode the input text
    text_embedding = model.encode([text])[0]
    
    # Get embeddings for all classes
    class_embeddings = compute_embeddings()
    
    # Calculate distances and return results
    results = []
    for class_name, class_embedding in class_embeddings:
        # Compute cosine similarity between text and class embedding
        similarity = np.dot(text_embedding, class_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(class_embedding))
        results.append({
            "class": class_name,
            "similarity": float(similarity)  # Convert tensor to float for JSON serialization
        })
    
    # Sort by similarity score descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results