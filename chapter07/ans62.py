import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def main():
    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)

    X_train_dicts = [d['feature'] for d in train_data]
    y_train = [d['label'] for d in train_data]

    vectorizer = DictVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_dicts)

    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_vec, y_train)

    print('モデルの学習が完了しました。')

    with open('model.pkl', 'wb') as f1, open('vectorizer.pkl', 'wb') as f2:
        pickle.dump(model, f1)
        pickle.dump(vectorizer, f2)
    
    print('モデルとベクトライザーを保存しました。')

if __name__ == "__main__":
    main()