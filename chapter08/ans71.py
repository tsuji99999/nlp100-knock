import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] 
SST_DIR = BASE_DIR / 'chapter07' / 'SST-2'

TRAIN_PATH = SST_DIR / 'train.tsv'
DEV_PATH = SST_DIR / 'dev.tsv'

def word_to_id(text, word2id):
    """テキストをID列(テンソル)に変換する"""
    words = text.split()
    input_ids = []
    for word in words:
        if word in word2id:
            input_ids.append(word2id[word])
        else:
            continue # 未知語は無視
    return torch.tensor(input_ids)


def main():
    # 訓練セット、開発セットを読み込む
    with open(TRAIN_PATH, 'r') as f1, open(DEV_PATH, 'r') as f2:
        train_data = pd.read_csv(f1, sep='\t')
        dev_data = pd.read_csv(f2, sep='\t')
    
    print(f'train_data length: {len(train_data)}')
    print(f'dev_data length: {len(dev_data)}')
    
    # 単語とIDの対応を読み込む
    with open('token_mapping.pkl', 'rb') as f:
        token_mapping = pickle.load(f)
    word2id = token_mapping['word_to_id']

    # トークンID列を格納するためのリスト
    train_dicts = []
    dev_dicts = []

    # 訓練セットのデータを変換
    for row in train_data.itertuples(index=False):
        text = row.sentence
        label = torch.tensor(row.label, dtype=torch.float32)
        input_ids = word_to_id(text, word2id)
        if len(input_ids) == 0:
            continue  # トークンID列が空の場合はスキップ
        train_dict = {
            'text': text,
            'label': label,
            'input_ids': input_ids
        }
        train_dicts.append(train_dict)
    
    # 開発セットのデータを変換
    for row in dev_data.itertuples(index=False):
        text = row.sentence
        label = torch.tensor(row.label, dtype=torch.float32)
        input_ids = word_to_id(text, word2id)
        if len(input_ids) == 0:
            continue  # トークンID列が空の場合はスキップ
        dev_dict = {
            'text': text,
            'label': label,
            'input_ids': input_ids
        }
        dev_dicts.append(dev_dict)
    
    print(f'Train data size: {len(train_dicts)}')
    print(f'Dev data size: {len(dev_dicts)}')

    # 保存
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_dicts, f)
    with open('dev_data.pkl', 'wb') as f:
        pickle.dump(dev_dicts, f)
    print('Data saved to train_data.pkl and dev_data.pkl')


if __name__ == "__main__":
    main()