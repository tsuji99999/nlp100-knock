import pickle
from sklearn.metrics import accuracy_score

def main():
    # モデル, ベクトライザー, 検証データの読み込み
    with open('model.pkl', 'rb') as f1, open('vectorizer.pkl', 'rb') as f2, open('dev_data.pkl', 'rb') as f3:
        model = pickle.load(f1)
        vectorizer = pickle.load(f2)
        dev_data = pickle.load(f3)

    X_dev_dicts = [d['feature'] for d in dev_data]
    y_dev = [d['label'] for d in dev_data]

    # 検証データのベクトル化
    X_dev_vec = vectorizer.transform(X_dev_dicts)

    # 予測
    y_pred = model.predict(X_dev_vec)
    
    # 正解率の算出
    accuracy = accuracy_score(y_dev, y_pred)
    print(f'検証データに対する正解率: {accuracy:.4f}')


if __name__ == "__main__":
    main()