import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
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
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)

    # t-SNE
    X = np.array(vectors)
    countries_tsne = tsne.fit_transform(X)

    # L2ノルムで正規化を行う
    X_normalized = normalize(X, norm='l2')

    # KMeansの実行
    kmeans_model = KMeans(n_clusters=5, random_state=10)
    kmeans_model.fit(X_normalized)

    # kmeansの結果で色分けしてt-SNEを表示
    plt.figure(figsize=(8, 8))
    plt.scatter(countries_tsne[:, 0], countries_tsne[:, 1], c=kmeans_model.labels_, cmap='viridis')

    for i, country in enumerate(valid_countries):
        plt.annotate(country, xy=(countries_tsne[i, 0], countries_tsne[i, 1]), fontsize=9)
    
    plt.title("t-SNE Visualization of Country Vectors")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    plt.savefig('ans59_tsne_countries.png')


if __name__ == "__main__":
    main()