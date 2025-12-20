#feature engineering module
#raw text into nos

import re

def extract_metadata_features(text):
    length = len(text)

    exclamations = text.count('!')
    capitals = sum(1 for c in text if c.isupper())
    letters = sum(1 for c in text if c.isalpha())

    capital_ratio = capitals / letters if letters > 0 else 0

    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = len(set(words))
    total_words = len(words)

    unique_word_ratio = unique_words / total_words if total_words > 0 else 0

    return [
        length,
        exclamations,
        capital_ratio,
        unique_word_ratio
    ]

#o/p:[234, 3, 0.12, 0.65] #length, exclamations, capital_letters, unique_words