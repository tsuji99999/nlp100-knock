import pickle
import torch
import numpy as np
import torch.nn as nn
from ans72 import LogisticRegression


def mean_embedding(input_ids: torch.Tensor,
                   embedding_matrix: torch.Tensor) -> torch.Tensor:
    """
    入力ID列に対応する単語埋め込みの平均ベクトルを計算する
    
    Args:
        input_ids (torch.Tensor): トークンID列 (文の長さ,)
        embedding_matrix (torch.Tensor): 単語埋め込み行列 (語彙数, 埋め込み次元数)
    Returns:
        torch.Tensor: 平均ベクトル (埋め込み次元数,)
    """
    # id列に対応する埋め込みを取り出して、0次元で平均をとる
    sentence_embs = embedding_matrix[input_ids] # shape=(文の長さ, 埋め込み次元数)
    mean_vector = sentence_embs.mean(dim=0) # shape=(埋め込み次元数,)
    return mean_vector

def train_model(
        model: LogisticRegression,
        X_train: list[torch.Tensor],
        y_train: list[torch.Tensor],
        num_epochs: int = 10,
        learning_rate: float = 0.01,
) -> None:
    """
    ロジスティック回帰モデルを学習する

    Args:
        model (LogisticRegression): 学習するモデル
        X_train torch.Tensor: 入力特徴量
        y_train torch.Tensor: 正解ラベル
        num_epochs (int): エポック数
        learning_rate (float): 学習率
    """
    criterion = nn.BCELoss()  # 二値交差エントロピー損失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(1, num_epochs+1):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch}: loss = {loss.item():.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train用のデータを読み込む
    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    # 単語埋め込み行列を読み込む
    embedding_matrix_np = np.load('embedding_matrix.npy')
    embedding_matrix = torch.tensor(embedding_matrix_np, dtype=torch.float32, device=device)
    embedding_dim = embedding_matrix.shape[1]

    # 学習データを用意
    X_list = []
    y_list = []

    # 各データについて平均ベクトルを求める
    for data in train_data:
        input_ids = data['input_ids'].to(device)
        mean_vector = mean_embedding(input_ids, embedding_matrix)
        X_list.append(mean_vector)

        label = data['label'].to(device)
        y_list.append(label)
    
    # バッチ用のテンソルに変換
    X_train = torch.stack(X_list, dim=0) # shape=(データ数, 埋め込み次元数)
    y_train = torch.stack(y_list, dim=0) # shape=(データ数,)

    # 形状を調整
    y_train = y_train.view(-1, 1) # shape=(データ数, 1)

    # モデルの作成
    model = LogisticRegression(embedding_dim=embedding_dim).to(device)
    print(model)
    
    # モデルの学習
    train_model(model, X_train, y_train, num_epochs=200, learning_rate=0.01)

    # モデルの保存
    torch.save(model.state_dict(), 'logistic_regression_model.pth')


if __name__ == "__main__":
    main()