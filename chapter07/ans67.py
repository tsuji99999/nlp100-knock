import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # モデル, ベクトライザーの読み込み
    with open('model.pkl', 'rb') as f1, open('vectorizer.pkl', 'rb') as f2:
        model = pickle.load(f1)
        vectorizer = pickle.load(f2)

    # 学習データ, 検証データの読み込み
    with open('train_data.pkl', 'rb') as f1, open('dev_data.pkl', 'rb') as f2:
        train_data = pickle.load(f1)
        dev_data = pickle.load(f2)
        
    # 学習データ, 検証データのリストを用意
    x_train_dicts = [d['feature'] for d in train_data]
    y_train = [d['label'] for d in train_data]
    X_dev_dicts = [d['feature'] for d in dev_data]
    y_dev = [d['label'] for d in dev_data]

    # 学習データ, 検証データのベクトル化
    X_train_vec = vectorizer.transform(x_train_dicts)
    X_dev_vec = vectorizer.transform(X_dev_dicts)

    # 予測
    y_train_pred = model.predict(X_train_vec)
    y_dev_pred = model.predict(X_dev_vec)

    # 学習データの評価指標を算出
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    # 検証データの評価指標を算出
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    dev_precision = precision_score(y_dev, y_dev_pred)
    dev_recall = recall_score(y_dev, y_dev_pred)
    dev_f1 = f1_score(y_dev, y_dev_pred)

    print(f'学習データ - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}')
    print(f'検証データ - Accuracy: {dev_accuracy:.4f}, Precision: {dev_precision:.4f}, Recall: {dev_recall:.4f}, F1-score: {dev_f1:.4f}')

if __name__ == "__main__":
    main()