import pickle
import pandas as pd

def main():
    # モデル, ベクトライザーの読み込み
    with open('model.pkl', 'rb') as f1, open('vectorizer.pkl', 'rb') as f2:
        model = pickle.load(f1)
        vectorizer = pickle.load(f2)

    # 特徴量の名前, 重みを取得する
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    print(weights.shape)

    # wordとweightの辞書を作り、データフレームに変換
    df = pd.DataFrame({
        "word": feature_names,
        "weight": weights
    })

    # 昇順にソート
    df_sorted = df.sort_values('weight')

    print('重みの低い特徴量 トップ20:')
    print(df_sorted.head(20))
    
    print('\n重みの高い特徴量トップ20:')
    print(df_sorted.tail(20)[::-1])

if __name__ == "__main__":
    main()