"""
本問は、「問題42において、実験設定を変化させると正解率が変化するかどうかを調べよ。」という問題である。
ここでは、各問の選択肢をシャッフルするshuffle_choices関数と、すべての問題の選択肢をDに入れ替えるmakea_answer_D関数を実装した。
"""

import os
import sys
import random
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 環境変数ファイルの読み込み
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class LLMClient:
    def __init__(self, model_name="gpt-5-mini"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def ask(self, prompt):
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt
        )
        return response

def create_prompt(question, choices):

    prompt = f"""
以下の問題に対して、4つの選択肢A~Dの中から最も適切なものを1つ選びなさい。
ただし、回答は選択肢のアルファベットのみを返すこと。（出力例: A）

# 問題
{question}

# 選択肢
- A: {choices[0]}
- B: {choices[1]}
- C: {choices[2]}
- D: {choices[3]}

# 回答
"""
    return prompt

def shuffle_choices(choices, answer):
    # 選択肢とインデックスのペアを作成
    indexed_choices = list(enumerate(choices))
    # シャッフルする
    random.shuffle(indexed_choices)
    # インデックスを削除した選択肢リストを作成
    shuffled_choices = [c[1] for c in indexed_choices]
    # 正解のインデックスを更新
    old_index = ord(answer) - ord("A")
    new_index = indexed_choices.index((old_index, choices[old_index]))
    new_answer = chr(new_index + ord("A"))

    return shuffled_choices, new_answer

def make_answer_D(choices, answer):
    if answer == 'D':
        return choices, answer
    else:
        # 選択肢とインデックスのペアを作成
        new_choices = choices.copy()
        # 正解のインデックスを取得
        answer_index = ord(answer) - ord("A")
        # 正解のインデックスのタプルと、インデックスが3のタプルを交換
        new_choices[answer_index], new_choices[3] = new_choices[3], new_choices[answer_index]

        return new_choices, "D"

def main():
    # タスクのCSVファイルをpandasで読み込む
    try:
        df = pd.read_csv("college_computer_science.csv")
        print("csvファイルの読み込みに成功しました。")
    except FileNotFoundError as e:
        print(f"ファイルが見つかりません。 → {e}")
        sys.exit(1)

    # LLMクライアントの初期化
    llm_client = LLMClient()

    correct = 0
    wrong = 0

    for idx, row in df.iterrows():

        choices = [row['選択肢A'], row['選択肢B'], row['選択肢C'], row['選択肢D']]

        # 選択肢の入れ替え
        new_choices, new_answer = make_answer_D(choices, row['正解'])

        prompt = create_prompt(row['問題文'], new_choices)

        output = llm_client.ask(prompt).output_text
        if output == new_answer:
            correct += 1
            print(f"{idx}: Correct. output: {output} answer: {new_answer}")
        else:
            wrong += 1
            print(f"{idx}: Wrong. output: {output} answer: {new_answer}")

    # 最終結果
    accuracy = correct / (correct + wrong) * 100
    print(f"correct  : {correct}")
    print(f"wrong    : {wrong}")
    print(f"accuracy : {accuracy}%")

if __name__ == "__main__":
    main()