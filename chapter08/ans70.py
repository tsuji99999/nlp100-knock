import pickle
import numpy as np
from gensim.models import KeyedVectors

EMBEDDING_PATH = '../chapter06/GoogleNews-vectors-negative300.bin'

def main():
    with open('word_embedding.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 単語埋め込み行列を取得
    matrix = model.vectors

    # パディングトークンを作成
    padding  = np.zeros(matrix.shape[1], dtype=matrix.dtype)

    # パディング行を行列の先頭に追加
    matrix_with_pad = np.vstack((padding, matrix))

    # 単語とインデックスの対応を整理
    id_to_word = ['<PAD>'] + model.index_to_key
    word_to_id = {'<PAD>': 0}
    word_to_id.update({word: idx+1 for word, idx in model.key_to_index.items()})

    print(matrix_with_pad.shape)
    print(len(id_to_word))
    print(len(word_to_id))

    np.save('embedding_matrix.npy', matrix_with_pad)
    with open('token_mapping.pkl', 'wb') as f:
        pickle.dump({'word_to_id': word_to_id, 'id_to_word': id_to_word}, f)

if __name__ == "__main__":
    main()
