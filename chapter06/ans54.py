from gensim.models import KeyedVectors

# セクションを抽出する関数
def extract_section(data, section_name):
    section_header = f': {section_name}'
    start_idx = None

    for i, line in enumerate(data):
        stripped = line.strip()

        if stripped == section_header:
            start_idx = i + 1
            continue

        if start_idx is not None:
            if stripped.startswith(':') and stripped != section_header:
                return data[start_idx:i]
            if stripped == '':
                return data[start_idx:i]

    return data[start_idx:] if start_idx is not None else []

def main():
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    with open('questions-words.txt', 'r') as f:
        raw_data = f.readlines()

    capital_countries_data = extract_section(raw_data, "capital-common-countries")

    # 各事例に対して類似度の計算、記録
    with open('ans54_prediction.txt', 'w')as f:
        for line in capital_countries_data:
            words = line.split()
            most_similar_word, similarity = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)[0]
            f.write(f"{words[0]:<15}{words[1]:<15}{words[2]:<15}{words[3].strip():<15}{most_similar_word:<25}\t{similarity}\n")

if __name__ == "__main__":
    main()