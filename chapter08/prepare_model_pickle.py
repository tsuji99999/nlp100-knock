import pickle
from gensim.models import KeyedVectors

EMBEDDING_PATH = '../chapter06/GoogleNews-vectors-negative300.bin'

def main():
    model = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=True)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()