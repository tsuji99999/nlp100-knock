import pickle
from collections import Counter

TRAIN_PATH = 'SST-2/train.tsv'
DEV_PATH = 'SST-2/dev.tsv'


def bag_of_words(sentence):
    words = sentence.split()
    word_counts = dict(Counter(words))
    return word_counts

def main():
    with open(TRAIN_PATH, 'r') as f1, open(DEV_PATH, 'r') as f2:
        train_data = f1.readlines()[1:]
        dev_data = f2.readlines()[1:]
    
    train_list = []
    dev_list = []
    
    for line in train_data:
        bow_dict = {}
        splitted = line.strip().split(sep='\t')

        bow_dict['text'] = splitted[0]
        bow_dict['label'] = int(splitted[1])
        bow_dict['feature'] = bag_of_words(splitted[0])

        train_list.append(bow_dict)
    
    for line in dev_data:
        bow_dict = {}
        splitted = line.strip().split(sep='\t')

        bow_dict['text'] = splitted[0]
        bow_dict['label'] = int(splitted[1])
        bow_dict['feature'] = bag_of_words(splitted[0])

        dev_list.append(bow_dict)
    
    # 確認用
    print('学習データ 最初の事例:')
    print(train_list[0])
    print('検証データ 最初の事例:')
    print(dev_list[0])

    # データをファイルに保存 (62番以降の問題のため)
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_list, f)
    
    with open('dev_data.pkl', 'wb') as f:
        pickle.dump(dev_list, f)

if __name__ == "__main__":
    main()