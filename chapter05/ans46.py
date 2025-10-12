import os
from dotenv import load_dotenv
from openai import OpenAI
from llm_client import LLMClient

# 環境変数ファイルの読み込み
load_dotenv()

def main():
    # LLMクライアントの初期化
    llm_client = LLMClient()

    prompt = """
以下のお題に沿った川柳を10個生成してください。
お題: 「自然言語処理 / NLP」
"""

    response = llm_client.ask(prompt)

    with open("46_output.txt", "w") as f:
        f.write(response.output_text)

    print(response.output_text)

if __name__ == "__main__":
    main()