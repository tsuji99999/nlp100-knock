import os
from dotenv import load_dotenv
from openai import OpenAI

# 環境変数ファイルの読み込み
load_dotenv()
    
def main():
    # LLMクライアントの初期化
    llm_client = LLMClient()

    prompt = "つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。"

    response = llm_client.ask(prompt)

    print(response.output_text)

if __name__ == "__main__":
    main()