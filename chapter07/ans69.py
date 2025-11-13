import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer


def main():
    # 学習データ, 検証データの読み込み
    with open('train_data.pkl', 'rb') as f1, open('dev_data.pkl', 'rb') as f2:
        train_data = pickle.load(f1)
        dev_data = pickle.load(f2)
    
    # 学習データ, 検証データの作成
    x_train_dicts = [d['feature'] for d in train_data]
    y_train = [d['label'] for d in train_data]
    X_dev_dicts = [d['feature'] for d in dev_data]
    y_dev = [d['label'] for d in dev_data]

    vectorizer = DictVectorizer()

    # 学習データ, 検証データのベクトル化
    X_train_vec = vectorizer.fit_transform(x_train_dicts)
    X_dev_vec = vectorizer.transform(X_dev_dicts)

    # 逆正則化係数のリスト
    C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    # 正解率を格納するリスト
    accuracies = []

    for c in C_list:
        # cの値ごとに学習を行う
        model = LogisticRegression(solver='liblinear', C=c)
        model.fit(X_train_vec, y_train)
        
        # 検証用のXに対して予測
        y_pred = model.predict(X_dev_vec)
    
        # 正解率の算出
        accuracy = accuracy_score(y_dev, y_pred)

        print(f'C={c:<7} | Accuracy = {accuracy}')
        accuracies.append(accuracy)
    
    file_name = 'ans69.png'
    plt.plot(C_list, accuracies, marker='o')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs Regularization Parameter C')
    plt.savefig(file_name)
    print(f'グラフを {file_name} に保存しました。')


if __name__ == "__main__":
    main()