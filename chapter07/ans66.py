import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix

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

    # 混同行列の表示
    cm = confusion_matrix(y_dev, y_pred)
    labels = ['Negative', 'Positive']
    df = pd.DataFrame(cm, index=labels, columns=labels)
    print('Confusion Matrix:')
    print(df)


if __name__ == "__main__":
    main()