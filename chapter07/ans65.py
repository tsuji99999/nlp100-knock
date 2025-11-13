import pickle
from ans61 import bag_of_words

TEXT_FOR_ANALYZE = "the worst movie i 've ever seen"

def predict_sentiment(text, model, vectorizer):
    # Bag_of_Wordsの取得
    bow = bag_of_words(text)

    # ベクトル化
    X_pred = vectorizer.transform([bow])

    # 予測
    probabilities = model.predict_proba(X_pred)[0]

    # 予測のラベル (確率が最大のインデックス)
    y_pred = probabilities.argmax()

    if y_pred == 0:
        label = 'Negative'
        confidence = probabilities[0]
    else:
        label = 'Positive'
        confidence = probabilities[1]

    return label, confidence

def main():
    # モデル, ベクトライザーのロード
    with open('model.pkl', 'rb') as f1, open('vectorizer.pkl', 'rb') as f2:
        model = pickle.load(f1)
        vectorizer = pickle.load(f2)
    
    # ポジネガ予測
    label, confidence = predict_sentiment(TEXT_FOR_ANALYZE, model, vectorizer)
    
    print(f'入力テキスト: {TEXT_FOR_ANALYZE}')
    print(f'予測結果: {label}')
    print(f'確信度: {confidence:.2%}')
    
if __name__ == "__main__":
    main()  