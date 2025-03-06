import pandas as pd
import requests

def load_stories():
    # You can use any of these sources:
    # 1. Quotes API: https://api.quotable.io/quotes?limit=150
    # 2. Short stories dataset from Kaggle
    # 3. Reddit writing prompts
    # For this example, let's use quotes
    
    response = requests.get('https://api.quotable.io/quotes?limit=150')
    quotes = response.json()['results']
    
    text_samples = []
    for quote in quotes:
        text_samples.append(quote['content'])
    
    return text_samples

stories = load_stories()