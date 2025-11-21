import torch
import pickle
import numpy as np
from ans72 import LogisticRegression
from ans73 import mean_embedding

def main():
    # dev用のデータを読み込む
    with open('dev_data.pkl', 'rb') as f:
        dev_data = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 単語埋め込み行列を読み込む
    embedding_matrix_np = np.load('embedding_matrix.npy')
    embedding_matrix = torch.tensor(embedding_matrix_np, dtype=torch.float32, device=device)
    embedding_dim = embedding_matrix.shape[1]
    
    # 学習済みモデル構築して、重みをロードする
    model = LogisticRegression(embedding_dim=embedding_dim).to(device)
    state_dict = torch.load('logistic_regression_model.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # 一応評価モードに切り替えておく

    X_list = []
    y_list = []

    # 各データについて平均ベクトルを求める
    for data in dev_data:
        input_ids = data['input_ids'].to(device)
        mean_vector = mean_embedding(input_ids, embedding_matrix)
        X_list.append(mean_vector)

        label = data['label'].to(device)
        y_list.append(label)

    # 検証用データで予測
    X_dev = torch.stack(X_list, dim=0) # shape=(データ数, 埋め込み次元数)
    y_dev = torch.stack(y_list, dim=0) # shape=(データ数,)
    y_dev = y_dev.view(-1, 1) # shape=(データ数, 1)

    with torch.no_grad():
        outputs = model(X_dev)
        preds = (outputs >= 0.5).float()
        accuracy = (preds == y_dev).float().mean().item()

    print(f'Dev set accuracy: {accuracy:.4f}')

      
if __name__ == "__main__":
    main()