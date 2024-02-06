from gensim.models import Word2Vec
import pandas as pd

reviews = pd.read_csv('../../data_naver/cleaned_data.csv')

tokens = []
for sentence in reviews.reviews:
    token = sentence.split()
    tokens.append(token)
print(tokens[0])

embedding_model = Word2Vec(tokens, vector_size=100, window=4, min_count=20,
                           workers=4, epochs=100, sg=1)
embedding_model.save('../models/word2vec_restaurant_review.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))