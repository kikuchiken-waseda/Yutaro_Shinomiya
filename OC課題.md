# オープンキャンパス
# 実習まとめ
ChatGPTとマルチLLMエージェントシステム(以下MLAS)を活用した人狼系ゲームワードウルフを作成した。

## 概要
ChatGPTによるマルチエージェントシステムにより、キャラクターを与えられたキャラクターたちと会話しながら進行するワードウルフ型ゲームを制作・展示。

来場者は観客として、誰が“ウルフ”かを推理する。

## ルール説明動画
https://youtu.be/O3nMihNYKMs  
ルール説明に使った動画。当日プレイ前に視聴してもらった。

## 展示品解説スライド
https://www.canva.com/design/DAGtXTsBNb4/V5BikTNLeDKxN8-zmnDSFQ/edit  
LLMやMLAS、展示品の説明などをするために本番で実際に使ったスライド。

## MLAS（マルチLLMエージェントシステム）
- **MLAS**
複数のLLM（大規模言語モデル）を「エージェント」として動かし、各エージェントに役割・性格・目的を与え、相互作用を通じてタスクを遂行させるシステム。

- **今回作ったゲーム：ワードウルフ**
  - エージェント：美月、匠、彩花、大輝の4人
  - ユーザー：人間プレイヤーも1エージェント扱い
  - 役割・目的：
    - 少数派（ワードウルフ） → 正体を隠して自然に話す
    - 多数派（村人） → 会話を自然に広げつつ少数派を見抜く
  - 性格プロンプト：各エージェントの発言の口調・思考パターンを制御
  - 記憶（Memory）：過去発言を保存し、次の発言生成に反映

### ゲーム内フローとMLASの役割
- **お題の生成**
  - GPTにテーマとワード（多数派・少数派）を生成
  - MLAS視点：
    - システム全体で共通の環境変数（テーマ・ワード）を共有
    - 各エージェントは自分のワードだけを知る → **非対称情報を再現**

- **発言生成**
  - **agent_speak**で各エージェントが発言
  - MLAS視点：
    - 独立したエージェントとして動作：
      - 自分のワード（多数派／少数派）を踏まえた発言を生成
      - 会話履歴（直近3ターン）を入力として文脈を理解
    - システムプロンプトに従い立ち回り：
      - 少数派は疑われないように多数派の会話に合わせる
      - 多数派は自然に話しつつ少数派を推理
    - メモリ活用：
      - 過去発言を参照して一貫した発言を生成

### ゲーム内フローとMLASの役割
- **独立したエージェント設計**
  - 個別プロンプト＋個別メモリで性格・立ち回りを制御
- **非対称情報の保持**
  - 少数派だけ自分のワードが異なることを知る
- **会話履歴の共有と参照**
  - 直近の会話を文脈として取り込む → 会話の一貫性を確保
- **分散推論**
  - 各エージェントが独自に少数派を推理
  - 最終的に票を集計して勝敗判定
- **動的順番制御**
  - 話す順番をランダム化し、会話の自然さを演出
- **ユーザー介入の統合**
  - 人間プレイヤーも1エージェントとしてMLASフローに参加


## 使用技術と構成

本プロジェクトでは、Pythonを用いたバックエンドとHTMLによるフロントエンドを組み合わせ、OpenAIのAPIとLangChainを活用してワードウルフゲームの対話システムを構築した。エージェントは合計4体で、それぞれ異なるキャラクター設定と記憶を持ち、プレイヤー（人間）1名とともに自然な会話を展開しながら推理を行う。

### 実装コード
  - バックエンド
<pre><code>
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
import os
import random
import re
from dotenv import load_dotenv
from collections import Counter
import json

# .envファイルから環境変数を読み込む
load_dotenv()

# APIキー設定
os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")

# 共通のシステムプロンプト
common_setup = (
    "これからワードウルフをプレイします。\n"
    "以下のプレイヤー（エージェント4人＋あなた）が順番に発言し、最後に誰が少数派（ワードウルフ）かを当てます。\n"
    "出力はただ発言だけ（台本形式ではなく）でお願いします。\n"
    "多数派の言葉にできるだけ合わせ、矛盾のない自然な発言を行ってください。\n"
    "キーワードを直接言うと怪しまれます。周辺情報や曖昧な言い回しでごまかしてください。\n"
    "他人の発言に乗っかる、相槌を打つ、同意するなど、疑われない工夫をしてください。\n"
    "なるべく自然な口語調にしてください。近しい間柄のように敬語は使わないようにしてください。\n"
    "発言は一度の発言で全体で30〜60字にしてください。冗長にならないよう注意しつつ、話題から逸れないようにしてください。\n"
    "あなたにはお題の言葉が1つ与えられていますが、それが他の人と同じかどうかはわかりません。\n"
    "あなたの目的は“疑われないこと”です。できるだけ他のプレイヤーの雰囲気や方向性に合わせて発言してください。\n"
    "自然な話題の広げ方、ぼかした表現、当たり障りのない感想を用いてください。多数派の話題から外れすぎると疑われます。\n"
    "他のプレイヤーの発言を観察して、誰が仲間（多数派）で誰が少数派かを推理してください。\n"
    "似た話題をしている人は仲間の可能性が高いです。\n"
    "あなたが話題から外れていると感じたら、自分が少数派かもしれません。\n"
    "その場合は、多数派に合わせてうまく話題をすり合わせるか、質問で探るなどして立ち回ってください。\n"
    "序盤ではキーワードをぼかしたり、具体例でぼかして話すようにしましょう。\n"
    "\n=== 会話例1 ===\n"
    "お題（多数派）: 電車\nお題（少数派）: 飛行機\n"
    "美月: 混んでてもつい窓の外を見ちゃうんだよね。\n"
    "匠: 音の揺れが心地いいんだよな、読書がはかどる。\n"
    "彩花: わかるー、ぼーっと外見てるのが癒し！\n"
    "大輝: 出発時間ぴったりなのがすごいと思う！\n"
    "解説: 匠は「飛行機」だが「混んでいる」という言葉から自分が少数派尚ではないかと推測し、座席や読書の話に寄せてブレを抑えている。\n"
    "=== 会話例2 ===\n"
    "お題（多数派）: 紅茶\nお題（少数派）: コーヒー\n"
    "美月: 朝より午後に飲むことが多いかな。\n"
    "匠: 苦味より香りで選びたくなるね。\n"
    "彩花: ミルク入れる派？私はレモンかな〜。\n"
    "大輝: お菓子との相性が抜群だよね！\n"
    "解説: 匠は「コーヒー」だが「午後」という言葉と「コーヒー」が結び付かず、「香り」と他の飲み物と共有できる軸にして調整している。\n"
    "=== 会話例3 ===\n"
    "お題（多数派）: 猫\nお題（少数派）: 犬\n"
    "美月: 丸くなって寝てる姿に癒される〜。\n"
    "匠: 構ってくれるけど、ベタベタしすぎないとこが好き。\n"
    "彩花: あのツンデレ感がたまらん！\n"
    "大輝: 時々すり寄ってくる感じがかわいいよね！\n"
    "解説: 匠は「犬」だが、自分が多数派とは限らないので「構ってくれるけど距離感がある」という抽象的な表現を用いて当たり障りない中でなるべく具体的な発言をして場を濁している。\n"
)

def create_character(name, persona):
    memory = ConversationBufferMemory(return_messages=True)
    system = SystemMessage(
        content=(
            f"あなたは『{name}』というキャラクターです。性格: {persona}。\n"
            "この性格に合った口調、言葉選び、話し方を意識して発言してください。\n"
            f"{common_setup}"
        )
    )
    return {
        "name": name,
        "persona": persona,
        "memory": memory,
        "system": system,
        "weight": 1.0
    }

agents = [
    create_character("美月", "優等生で内気だが観察力が高い"),
    create_character("匠", "正義感が強く論理的"),
    create_character("彩花", "明るく社交的なムードメーカー"),
    create_character("大輝", "陽気で場を和ませるスポーツマン"),
]

user_name = "あなた"

def propose_topic_and_words():
    prompt = (
        "ワードウルフのテーマを一つ決めてください。"
        "さらに『多数派ワード』と『少数派ワード』をJSON形式で返してください。\n"
        "例: {\"theme\":\"果物\",\"major\":\"りんご\",\"minor\":\"バナナ\"}"
    )
    resp = llm.invoke([SystemMessage(content=prompt)])
    return resp.content.strip()

def select_wolf(names):
    return random.choice(names)

def get_agent_by_name(name):
    return next((a for a in agents if a["name"] == name), None)

def decide_next_speaker_from_context(agent_memories, spoken_names, candidates):
    remaining = [n for n in candidates if n not in spoken_names]
    if not remaining:
        last_speaker = list(spoken_names)[-1] if spoken_names else None
        available = [n for n in candidates if n != last_speaker]
        return random.choice(available)
    return random.choice(remaining)

def agent_speak(agent, word, context_names, utterances):
    recent_context = "\n".join([f"{utt['name']}: {utt['text']}" for utt in utterances[-3:]])
    agent["memory"].chat_memory.add_user_message(f"お題の言葉は「{word}」です。発言してください。")
    messages = [agent["system"]] + agent["memory"].chat_memory.messages + [
        HumanMessage(content=(
            f"直前の会話の流れ：\n{recent_context}\n"
            "今からあなたはこの流れを受けて一言だけ自然に返答してください。\n"
            "・発言は60字以内に抑えてください。\n"
            "・周囲の話題から逸れないようにしてください。\n"
            "・他人の発言に乗る、具体例や体験を挙げる、自然な相槌などを使ってください。\n"
            "・発言の主張がぼやけないようにしてください（曖昧な言い回しの多用を避けてください）。\n"
            "与えられたキーワードは使わず、それに関係ある発言をしてください。"
        ))
    ]
    resp = llm.invoke(messages)
    text = resp.content.strip()
    agent["memory"].chat_memory.add_ai_message(text)
    return text

def agent_vote(agent, utterances, names):
    prompt = (
        "以下はワードウルフの会話ログです。\n"
       "あなたはこの会話に参加しているプレイヤーの1人です。\n"
        "他のプレイヤーの発言を観察しながら、自分に与えられたお題と食い違っている可能性がある人物を推理してください。\n"
        "\n"
        "▼ 推理の観点：\n"
        "・誰かの発言が自分のお題では使いにくい／思いつかないような内容で構成されていないか？\n"
        "・誰かの語彙、描写、例えが自分のお題と異なる領域やジャンルに見えないか？\n"
       "・具体性の有無や話題の方向性がずれていないか？\n"
       "\n"
       "▼ 状況に応じた行動例：\n"
        "1. 自分が多数派だと思う → 少数派を見つけて票を入れる。\n"
        "2. 自分が少数派だと思う → 多数派の一人に疑いが向くように票を入れる。\n"
        "3. 自分の立場が曖昧 → 一番自分と話題のずれを感じる相手に票を入れる。\n"
        "\n"
        "▼ 注意点：\n"
        "・発言内容が抽象的すぎる／明確すぎるなど、語調や態度の違いもヒントになります。\n"
        "・文脈上、違和感があってもお題の範囲内に収まるかどうかを慎重に判断してください。\n"
        "\n"
        "【出力形式（厳守）】\n"
        "以下のJSON形式でのみ回答してください。\n"
        "{\"vote\": \"◯◯\", \"reason\": \"〜〜〜（80字以内）\"}\n"
        "\n"
        "=== 会話ログ ==="
    )

    for utt in utterances:
        prompt += f"\n{utt['name']}: {utt['text']}"

    messages = [SystemMessage(content="あなたは投票前に一言だけ全体の印象を述べてください（例：〇〇が少しずれている気がする）"),
                HumanMessage(content="全体の印象を一言で：")]
    llm.invoke(messages)  # 印象出力（捨てている）

    vote_model = ChatOpenAI(temperature=0.7, model_name="gpt-4")  # memoryなしの新しいインスタンス
    messages = [agent["system"], HumanMessage(content=prompt)]
    resp = vote_model.invoke(messages)
    response = resp.content.strip()

    try:
        parsed = json.loads(response)
        guess = parsed.get("vote")
        reason = parsed.get("reason")
        if guess not in names:
            guess = random.choice([n for n in names if n != agent["name"]])
    except json.JSONDecodeError:
        guess = random.choice([n for n in names if n != agent["name"]])
        reason = "形式不正のためランダム選択されました。"

    return guess, f"投票先：{guess}\n理由：{reason}"

def main():
    topic_json = propose_topic_and_words()
    import json
    try:
        topic = json.loads(topic_json)
    except json.JSONDecodeError:
        print("お題の読み取りに失敗しました。もう一度お試しください。")
        return
    theme = topic["theme"]
    major = topic["major"]
    minor = topic["minor"]

    names = [a["name"] for a in agents] + [user_name]
    wolf = select_wolf(names)
    words = {name: (minor if name == wolf else major) for name in names}

    agent_memories = {a["name"]: a["memory"] for a in agents}

    print(f"=== ワードウルフ開始 ===")
    print(f"あなたのお題は『{words[user_name]}』です。")

    turns = 10
    utterances = []
    spoken_names = set()
    next_speaker = random.choice(names)

    for _ in range(turns):
        word = words[next_speaker]
        if next_speaker == user_name:
            print(f"あなたのお題は『{word}』です。発言の中では明言しないでください。")
            try:
                inp = input(f"{next_speaker}（あなた）、発言を入力してください：")
            except EOFError:
                print("入力が中断されました。プログラムを終了します。")
                return
            print(f"{next_speaker}: {inp}")
            utterances.append({"name": next_speaker, "text": inp})
        else:
            agent = get_agent_by_name(next_speaker)
            text = agent_speak(agent, word, names, utterances)
            print(f"{agent['name']}: {text}")
            utterances.append({"name": agent["name"], "text": text})

        spoken_names.add(next_speaker)
        next_speaker = decide_next_speaker_from_context(agent_memories, spoken_names, names)

    print("=== 推理フェーズ ===")
    votes = []
    for agent in agents:
        guess, reason = agent_vote(agent, utterances, names)
        print(f"{agent['name']} ：{reason}")
        votes.append(guess)

    your_vote = input("\nあなたの投票：誰がワードウルフだと思いますか？名前を入力してください：").strip()
    votes.append(your_vote)

    print("\n=== 投票結果 ===")
    counter = Counter(votes)
    for name, count in counter.items():
        print(f"{name}: {count}票")

    winner = counter.most_common(1)[0][0]
    print(f"\n多数決で選ばれたワードウルフは「{winner}」です。")
    print(f"お題（多数派）: {major}")
    print(f"お題（少数派）: {minor}")

    if winner == wolf:
        if user_name == wolf:
            print("😢 あなたはワードウルフでしたが、見破られてしまいました。村人の勝ちです。")
        else:
            print("🎉 正解！ワードウルフを当てました！")
    else:
        if user_name == wolf:
            print("🎉 あなたはワードウルフでしたが、逃げ切りました！あなたの勝ちです！")
        else:
            print(f"😢 残念…正解は「{wolf}」でした。ワードウルフの勝ちです。")

if __name__ == "__main__":
    main() </code></pre>
