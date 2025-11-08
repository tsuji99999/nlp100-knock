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

# 正解率を算出する関数
def calc_accuracy(data, model):
    correct = 0
    wrong = 0

    for line in data:
        words = line.split()
        most_similar_word = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)[0][0]
        
        if words[3] == most_similar_word:
            correct += 1
        else:
            wrong += 1
    
    return correct / (correct + wrong) if (correct + wrong) > 0 else 0


def main():
    semantic_analogy = [
        'capital-common-countries',
        'capital-world',
        'currency',
        'city-in-state',
        'family',
    ]

    syntactic_analogy = [
        'gram1-adjective-to-adverb',
        'gram2-opposite',
        'gram3-comparative',
        'gram4-superlative',
        'gram5-present-participle',
        'gram6-nationality-adjective',
        'gram7-past-tense',
        'gram8-plural',
        'gram9-plural-verbs'
    ]

    # 生データを読み込む
    with open('questions-words.txt', 'r') as f:
        raw_data = f.readlines()

    # モデルのロード
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # 意味的アナロジーの正解率を測定
    for section_name in semantic_analogy:
        data = extract_section(raw_data, section_name)
        accuracy = calc_accuracy(data, model)
        print(f'{section_name}: {accuracy}')
    
    # 文法的アナロジーの正解率を測定
    for section_name in syntactic_analogy:
        data = extract_section(raw_data, section_name)
        accuracy = calc_accuracy(data, model)
        print(f'{section_name}: {accuracy}')

if __name__ == "__main__":
    main()