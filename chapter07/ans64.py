import pickle

def main():
    with open('model.pkl', 'rb') as f1, open('vectorizer.pkl', 'rb') as f2, open('dev_data.pkl', 'rb') as f3:
        model = pickle.load(f1)
        vectorizer = pickle.load(f2)
        dev_data = pickle.load(f3)

    X_first_dict = [dev_data[0]['feature']]

    # 検証データのベクトル化(先頭の1件のみ)
    X_first_vec = vectorizer.transform(X_first_dict)

    # 予測
    y_pred_probability = model.predict_proba(X_first_vec)
    print(f'先頭の事例が0(Negative)である条件付き確率: {y_pred_probability[0][0]}')
    print(f'先頭の事例が1(Positive)である条件付き確率: {y_pred_probability[0][1]}')


if __name__ == "__main__":
    main()