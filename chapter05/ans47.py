import os
from dotenv import load_dotenv
from openai import OpenAI

senryus = [
    "データ積み モデル育てる 朝が来る",
    "単語切り 改行も泣く 仕方ない",
    "ベクトルで 言葉が踊る 距離測る",
    "文を読む 曖昧ほどく 答え出す",
    "翻訳機 言を渡す 橋架ける",
    "バイアスが データに潜む 声あげよ",
    "プロンプト 魔法みたいだ 答え出す",
    "精度見る スコア踊れば 胸騒ぎ",
    "名前消す 記録残るよ 気をつけて",
    "会話する 言葉の余白 人の声"
]

def create_prompt(senryus):
    prompt_for_judge = """
# タスク説明
あなたは、川柳の専門家です。以下に示す10個の川柳の面白さを、それぞれ10段階で評価してください。

# 出力形式
1. (川柳1をここに記載)
    評価:[1~10の整数]
    理由:[評価理由の簡潔な説明]

2. (川柳2をここに記載)
    評価:[1~10の整数]
    理由:[評価理由の簡潔な説明]

(以下10個分続く)

# 評価対象
"""
    for i, senryu in enumerate(senryus):
        prompt_for_judge += f"{i+1}. {senryu}\n"
    
    return prompt_for_judge


def main():

    # 川柳の出力結果を読み込む
    with open('46_output.txt', 'r') as f:
        senryu = f.read()

    # 環境変数を読み込む
    load_dotenv()

    # プロンプトを作成
    prompt_for_judge = create_prompt(senryu)

    # 評価にGPT-4oを使用
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt_for_judge}
        ]
    )

    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()