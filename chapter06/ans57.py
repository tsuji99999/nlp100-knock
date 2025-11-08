from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
from ans54 import extract_section
from gensim.models import KeyedVectors

DATA_PATH = 'questions-words.txt'

def extract_countries(data):
    """スペース区切りで与えられたデータから国名のリストを抽出する関数"""
    countries = set()
    for line in data:
        words = line.split()
        countries.add(words[1])
        countries.add(words[3])
    return list(countries)

def main():
    # モデルのロード
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # データの読み込み
    with open(DATA_PATH, 'r') as f:
        raw_data = f.readlines()
    
    # capital-common-countries セクションの抽出
    capital_countries_data = extract_section(raw_data, "capital-common-countries")
    
    # 国名の抽出
    countries = extract_countries(capital_countries_data)

    valid_countries = []
    vectors = []

    # 入力データの用意
    for country in countries:
        if country in model:
            valid_countries.append(country)
            vectors.append(model[country])
    X = np.array(vectors)

    print(f"対象国数: {len(valid_countries)}")
    print(f"入力データの形状: {X.shape}")

    # L2ノルムで正規化を行う
    X_normalized = normalize(X, norm='l2')

    # KMeansの実行
    kmeans_model = KMeans(n_clusters=5, random_state=10)
    kmeans_model.fit(X_normalized)

    for i in range(5):
        cluster = np.array(valid_countries)[kmeans_model.labels_ == i]
        print(f"Cluster {i}: {cluster}")


if __name__ == "__main__":
    main()