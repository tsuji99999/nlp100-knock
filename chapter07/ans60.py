import pandas as pd

TRAIN_DATA_PATH = 'SST-2/train.tsv'
DEV_DATA_PATH = 'SST-2/dev.tsv'

def main():
    # データの読み込み
    df_train = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
    df_train_label = df_train["label"]
    df_dev = pd.read_csv(DEV_DATA_PATH, sep='\t')
    df_dev_label = df_dev["label"]

    # ラベルの分布を表示
    print('学習データのラベル分布:')
    print(df_train_label.value_counts())
    print('\n検証データのラベル分布:')
    print(df_dev_label.value_counts())

if __name__ == "__main__":
    main()