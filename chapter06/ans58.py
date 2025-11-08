import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import linkage, dendrogram
from ans54 import extract_section
from ans57 import extract_countries

DATA_PATH = 'questions-words.txt'

def main():
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # データの読み込み
    with open(DATA_PATH, 'r') as f:
        raw_data = f.readlines()

    # 国名に関するデータを抽出
    capital_countries_data = extract_section(raw_data, "capital-common-countries")

    # 国のリストを作成
    countries = extract_countries(capital_countries_data)

    # 各国のベクトルを用意
    valid_countries = []
    vectors = []
    for country in countries:
        if country in model:
            valid_countries.append(country)
            vectors.append(model[country])

    df = pd.DataFrame(vectors, index=valid_countries)
    df.index.name = "Country"

    df_normalized = df.div(df.pow(2).sum(axis=1).pow(0.5), axis=0)
    
    # クラスタリング
    clusterd = linkage(df_normalized, method='ward', metric="euclidean")

    # デンドログラムの表示
    plt.figure(figsize=(8, 5), dpi=100, facecolor='c')
    dendrogram(clusterd, labels=valid_countries, leaf_font_size=8, orientation='right')
    plt.savefig('ans58_dendrogram_normalized.png')
    

if __name__ == "__main__":
    main()