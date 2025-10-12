import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 環境変数からAPIキーを読み込む
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt40 = """
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

# APIにリクエスト送信
try:
    response = client.responses.create(
        model='gpt-5-mini',
        input=prompt40
    )

    print(response.output_text)

except Exception as e:
    print(f"エラーが発生しました: {e}")