import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

DATA_PATH = 'wordsim353/combined.csv'

def main():
    # モデルのロード
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # WordSimilarity-353の評価データを読み込む
    df = pd.read_csv(DATA_PATH)

    # 1,2列目のコサイン類似度を算出
    df['word2vec_sim'] = df.apply(
        lambda row: model.similarity(row['Word 1'], row['Word 2']) if row['Word 1'] in model and row['Word 2'] in model else np.nan,
        axis=1
    )
    
    corr_spearman = df[["Human (mean)", "word2vec_sim"]].corr(method="spearman")
    print(corr_spearman)

if __name__ == "__main__":
    main()