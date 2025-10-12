import os
from dotenv import load_dotenv
from openai import OpenAI
from llm_client import LLMClient

# 環境変数ファイルの読み込み
load_dotenv()
    
def main():
    # LLMクライアントの初期化
    llm_client = LLMClient()

    # === 1ターン目 ===
    # 応答を生成する
    prompt1 = "つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。"
    response1 = llm_client.ask(prompt1).output_text
    print(response1)
    
    # === 2ターン目 ===
    # プロンプトを作成
    question = "さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？"
    prompt2 = "# 会話履歴\n私:\n" + prompt1 + "\nあなた(ChatGPT):\n" + response1 + "\n私:\n" + question + "\nあなた(ChatGPT):"
    print("送信されるプロンプト:\n", prompt2)

    # 応答を生成
    response2 = llm_client.ask(prompt2).output_text
    print(response2)


if __name__ == "__main__":
    main()