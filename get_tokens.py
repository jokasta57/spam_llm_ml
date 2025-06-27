'''

This code identifies the most important tokens in your medical text using a combined approach:

It tokenizes the text using the RoBERTa tokenizer
Cleans the tokens by removing special tokens and handling word pieces (removing the 'Ġ' prefix)
Applies TF-IDF analysis to identify statistically significant terms
Removes Spanish stopwords (common words like "de", "la", "el")
Calculates an importance score based on both frequency and TF-IDF values
Returns the top 10 most important tokens sorted by their importance score
When you run this code, it will output the original tokens from the tokenizer followed by the top 10 most important tokens ranked by their significance in the text. 
The tokens will likely include medical terms like "ataxia", "cerebelosa", "ATXN10", etc., which carry the most diagnostic value in this patient case description.

'''

from transformers import AutoTokenizer
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

def get_top_ten_tokens(input_text):
    
    # Load tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize text
    encoded_input = tokenizer(input_text)
    tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"])

    # Clean tokens (remove special tokens and word pieces)
    clean_tokens = []
    for token in tokens:
        # Skip special tokens and word pieces that start with Ġ
        if token not in tokenizer.all_special_tokens and not token.startswith('Ġ'):
            clean_tokens.append(token)
        # For tokens starting with Ġ, remove the Ġ prefix
        elif token.startswith('Ġ'):
            clean_tokens.append(token[1:])

    # Spanish stopwords
    spanish_stopwords = [
       "de", "la", "el", "y", "en", "que", "a", "los", "del", "se", "las", "por", "un", "para",
       "con", "una", "su", "al", "es", "lo", "como", "más", "pero", "sus", "le", "ya", 
       "o", "fue", "este", "ha", "si", "porque", "esta", "son", "cuando", "muy",
       "sin", "sobre", "ser", "también", "me", "hay", "donde", "quien", "desde",
       "nos", "todos", "uno", "les", "otros", "fueron", "ese", "eso",
       "puede", "primera", "mediante", "gran"
       ]

    # Define function to clean text for TF-IDF
    def clean_text(text):
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        return text

    # Process text for TF-IDF
    cleaned_text = clean_text(input_text)

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer(stop_words=spanish_stopwords)
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()

    # Get TF-IDF scores
    tfidf_scores = {}
    for col in range(tfidf_matrix.shape[1]):
       tfidf_scores[feature_names[col]] = tfidf_matrix[0, col]

    # Count token occurrences (after removing stopwords)
    filtered_tokens = [token.lower() for token in clean_tokens if token.lower() not in spanish_stopwords]
    token_count = Counter(filtered_tokens)

    # Calculate importance score using both TF-IDF and frequency
    importance_scores = {}
    for token, count in token_count.items():
       # Try to find token in TF-IDF scores
       if token in tfidf_scores:
           importance_scores[token] = tfidf_scores[token] * count
       else:
           # Fallback for tokens not in TF-IDF
           importance_scores[token] = count * 0.5

    # Get top 10 important tokens
    top_tokens = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    top_ten_tokens = []
    for i, (token, score) in enumerate(top_tokens, 1):
        top_ten_tokens.append(score)
    
    return top_ten_tokens