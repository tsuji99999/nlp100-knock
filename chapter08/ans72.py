import torch
import torch.nn as nn

# NNの設計
class LogisticRegression(nn.Module):
    def __init__(self, embedding_dim: int):
        """
        単語埋め込みの平均ベクトルを用いた分類器

        Args:
            embedding_dim (int): 単語埋め込みの次元数
        """
        super().__init__()
        self.l1 = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid() # 出力を0-1に変換
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        Args:
            x (torch.Tensor): 入力テンソル (バッチサイズ, 埋め込み次元数)
        Returns:
            torch.Tensor: 出力テンソル (バッチサイズ, 1)
        """
        out = self.l1(x)
        out = self.sigmoid(out)
        return out