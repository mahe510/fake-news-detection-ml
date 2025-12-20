from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity(input_vector, reference_vectors):
    similarities = cosine_similarity(input_vector, reference_vectors)
    max_similarity = np.max(similarities)
    return max_similarity

