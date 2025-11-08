from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

cosine_similarity = model.similarity('United_States', 'U.S.')

print(f"コサイン類似度: {cosine_similarity}")