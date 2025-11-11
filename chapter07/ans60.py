import pandas as pd

DATA_PATH = 'SST-2/train.tsv'

def main():
    
    df = pd.read_csv(DATA_PATH, sep='\t')
    df_label = df["label"]
    
    print(f"Positive: {(df_label == 1).sum()}")
    print(f"Negative: {(df_label == 0).sum()}")

if __name__ == "__main__":
    main()