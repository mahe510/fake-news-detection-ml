#“I built a Fake News Detection system using NLP and supervised machine learning,
#  trained on real-world data, achieving ~99% accuracy, with a live prediction pipeline"

#Improve Features with N-GRAMS=(1,2) (BIG UPGRADE):Use single words and word pairs
#Fake news often uses phrases, not just words: Unigrams (single words) miss this context.

#I evaluated my fake news classifier using precision, recall, F1-score, and confusion matrix.
#I further improved performance by adding n-grams and stopword removal to the TF-IDF pipeline.

from features import extract_metadata_features

text = "BREAKING!!! You WON 10 CRORE Lottery!!! CLICK NOW"
features = extract_metadata_features(text)

print(features)
