from transformers import DistilBertTokenizer, DistilBertModel

import torch
import numpy as np

# Load once globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

bert_model.to(device)
bert_model.eval()
def get_bert_embeddings(text_series):
    embeddings = []

    for text in text_series:
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128 
        )

        #inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = bert_model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        combined = hstack([text_vector, meta, cls_embedding]).tocsr()
        embeddings.append(cls_embedding.cpu().numpy().flatten())

    return np.array(embeddings)
