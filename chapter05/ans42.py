import os
import sys
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


def main():

    # タスクのCSVファイルをpandasで読み込む
    try:
        df = pd.read_csv("college_computer_science.csv")
        print("csvファイルの読み込みに成功しました。")
    except FileNotFoundError as e:
        print(f"ファイルが見つかりません。 → {e}")
        sys.exit(1)

    # LLMクライアントの初期化
    client = LLMClient()

    correct = 0
    wrong = 0

    for idx, row in df.iterrows():

        choices = [row['選択肢A'], row['選択肢B'], row['選択肢C'], row['選択肢D']]
        prompt = create_prompt(row['問題文'], choices)

        output = client.ask(prompt).output_text
        if output == row['正解']:
            correct += 1
            print(f"{idx}: Correct.")
        else:
            wrong += 1
            print(f"{idx}: Wrong. output: {output} answer: {row['正解']}")

    # 最終結果
    accuracy = correct / (correct + wrong) * 100
    print(f"correct  : {correct}")
    print(f"wrong    : {wrong}")
    print(f"accuracy : {accuracy}%")

if __name__ == "__main__":
    main()