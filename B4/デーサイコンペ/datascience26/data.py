"""
MeCab + 日本語評価極性辞書（東北大学 乾・岡崎研究室）による
新聞見出しのポジネガ判定

前提:
    pip3 install mecab-python3 ipadic pandas

    以下2つの辞書を、このスクリプトと同じフォルダに置く。
    どちらか一方だけでも動く。
      - wago.121808.pn            （用言編：動詞・形容詞）
      - pn.csv.m3.120408.trim     （名詞編）

    補助辞書 domain_pn.csv は任意。福島関連の重要語を補う。
      書式: 語,スコア,備考  （スコアは -1〜+1）

引用について:
    用言編を使う場合は 小林ら(2005)、
    名詞編を使う場合は 東山ら(2008) を論文・発表資料で引用すること。

入力:
    headlines.tsv … 列: 品目 / 紙名 / 発行日 / 年 / 見出し / 備考

出力:
    headlines_scored.tsv … 元データ + 判定結果
    yearly_sentiment.tsv … 年次サマリ
"""

import os
import sys
from collections import Counter

import MeCab
import ipadic
import pandas as pd

# ============================================================
# 設定
# ============================================================
INPUT = "headlines.tsv"
DIC_WAGO = "wago.121808.pn"           # 用言編
DIC_MEISHI = "pn.csv.m3.120408.trim"  # 名詞編
DOMAIN_DIC = "domain_pn.csv"          # 任意

OUT_SCORED = "headlines_scored.tsv"
OUT_YEARLY = "yearly_sentiment.tsv"

TARGET_POS = {"名詞", "動詞", "形容詞", "副詞"}

# 平均スコアがこの値を超えたらポジ、下回ったらネガ
THRESHOLD = 0.10

# 辞書ヒット語がこれ未満なら「判定不能」
MIN_HITS = 1

# 「コメ」検索で混入する非・稲作の語
NOISE_MARKERS = ["コメント", "コメリ", "コメディ", "コメて", "コメた"]

ENCODINGS = ("utf-8", "utf-8-sig", "euc_jp", "shift_jis", "cp932")

# ============================================================


def read_lines(path):
    """文字コードを自動判定して行のリストを返す"""
    for enc in ENCODINGS:
        try:
            with open(path, encoding=enc) as f:
                lines = f.read().splitlines()
            return lines, enc
        except UnicodeDecodeError:
            continue
    sys.exit(f"エラー: {path} の文字コードを判定できませんでした。")


def load_wago(path):
    """
    用言編を読み込む。
    形式: 極性ラベル <TAB> 語（形態素が空白区切り）
    例:   ポジ（評価）\t優れ る
    列順が逆の版もあるため自動判定する。
    """
    if not os.path.exists(path):
        print(f"用言編 {path} は未配置。スキップします。")
        return {}

    lines, enc = read_lines(path)
    table = {}
    swapped = 0

    for line in lines:
        parts = [p for p in line.split("\t") if p.strip()]
        if len(parts) < 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()

        # 極性ラベルがどちらの列にあるか判定
        if a.startswith(("ポジ", "ネガ")):
            label, word = a, b
        elif b.startswith(("ポジ", "ネガ")):
            label, word = b, a
            swapped += 1
        else:
            continue

        # 「優れ る」のように空白で区切られているので連結する
        word = word.replace(" ", "").replace("　", "")
        if not word:
            continue

        table[word] = 1.0 if label.startswith("ポジ") else -1.0

    note = "（列順を反転して読み込み）" if swapped > len(table) / 2 else ""
    print(f"用言編: {len(table)}語 を読み込みました（{enc}）{note}")
    return table


def load_meishi(path):
    """
    名詞編を読み込む。
    形式: 語 <TAB> 極性(p/n/e) <TAB> 説明
    列順が逆の版もあるため自動判定する。
    p=ポジティブ n=ネガティブ e=中立
    """
    if not os.path.exists(path):
        print(f"名詞編 {path} は未配置。スキップします。")
        return {}

    lines, enc = read_lines(path)
    table = {}
    mapping = {"p": 1.0, "n": -1.0, "e": 0.0}

    for line in lines:
        parts = [p for p in line.split("\t") if p.strip()]
        if len(parts) < 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()

        if b in mapping:
            word, label = a, b
        elif a in mapping:
            word, label = b, a
        else:
            continue

        word = word.replace(" ", "").replace("　", "")
        if word:
            table[word] = mapping[label]

    n_e = sum(1 for v in table.values() if v == 0.0)
    print(f"名詞編: {len(table)}語 を読み込みました（{enc}）／うち中立 {n_e}語")
    return table


def load_domain(path):
    if not os.path.exists(path):
        print(f"補助辞書 {path} は未配置。")
        return {}
    df = pd.read_csv(path)
    d = {str(r["語"]).strip(): float(r["スコア"]) for _, r in df.iterrows()}
    print(f"補助辞書: {len(d)}語 を読み込みました")
    return d


def tokenize(tagger, text):
    tokens = []
    node = tagger.parseToNode(str(text))
    while node:
        if node.surface:
            f = node.feature.split(",")
            pos = f[0]
            base = f[6] if len(f) > 6 and f[6] != "*" else node.surface
            tokens.append((node.surface, pos, base))
        node = node.next
    return tokens


def score_headline(tokens, wago, meishi, domain):
    """
    優先順位: 補助辞書 > 用言編 > 名詞編
    中立(0.0)の語もヒットとして数える（中立が多いこと自体が情報）
    """
    hits, detail = [], []

    for surface, pos, base in tokens:
        if pos not in TARGET_POS:
            continue

        for key in (base, surface):
            if key in domain:
                hits.append(domain[key])
                detail.append(f"{key}:{domain[key]:+.1f}(補助)")
                break
            if key in wago:
                hits.append(wago[key])
                detail.append(f"{key}:{wago[key]:+.1f}(用言)")
                break
            if key in meishi:
                hits.append(meishi[key])
                detail.append(f"{key}:{meishi[key]:+.1f}(名詞)")
                break

    if not hits:
        return 0.0, 0, 0, ""

    # 中立語を除いた極性語だけの数も返す（判定の信頼度の目安）
    polar = [h for h in hits if h != 0.0]
    return sum(hits) / len(hits), len(hits), len(polar), " ".join(detail)


def judge(score, n_polar):
    if n_polar < MIN_HITS:
        return "判定不能"
    if score > THRESHOLD:
        return "ポジティブ"
    if score < -THRESHOLD:
        return "ネガティブ"
    return "中立"


def is_noise(text):
    return any(m in str(text) for m in NOISE_MARKERS)


def main():
    if not os.path.exists(INPUT):
        sys.exit(f"エラー: {INPUT} が見つかりません。")

    wago = load_wago(DIC_WAGO)
    meishi = load_meishi(DIC_MEISHI)
    domain = load_domain(DOMAIN_DIC)

    if not wago and not meishi:
        sys.exit("エラー: 極性辞書が1つも読み込めませんでした。")

    tagger = MeCab.Tagger(ipadic.MECAB_ARGS)

    df = pd.read_csv(INPUT, sep="\t", dtype=str)
    if "見出し" not in df.columns:
        sys.exit(f"エラー: 見出し列がありません。列: {list(df.columns)}")

    print(f"\n入力: {len(df)}件\n判定中...\n")

    scores, n_hits, n_polars, details, judges, noises, toks = [], [], [], [], [], [], []

    for text in df["見出し"].fillna(""):
        tokens = tokenize(tagger, text)
        s, n, np_, detail = score_headline(tokens, wago, meishi, domain)
        scores.append(round(s, 4))
        n_hits.append(n)
        n_polars.append(np_)
        details.append(detail)
        judges.append(judge(s, np_))
        noises.append("要確認" if is_noise(text) else "")
        toks.append(" ".join(b for _, p, b in tokens
                             if p in TARGET_POS and len(b) > 1))

    df["スコア"] = scores
    df["辞書ヒット数"] = n_hits
    df["極性語数"] = n_polars
    df["判定"] = judges
    df["ノイズ"] = noises
    df["寄与語"] = details
    df["分かち書き"] = toks

    df.to_csv(OUT_SCORED, sep="\t", index=False, encoding="utf-8-sig")
    print(f"保存: {OUT_SCORED}")

    print("\n=== 判定の内訳 ===")
    for k, v in df["判定"].value_counts().items():
        print(f"  {k}: {v}件")
    print(f"  ノイズ要確認: {(df['ノイズ'] == '要確認').sum()}件")
    print(f"  極性語数の中央値: {df['極性語数'].median():.1f}語")

    # ---- 年次サマリ ----
    clean = df[df["ノイズ"] != "要確認"].copy()
    clean["年"] = clean["年"].astype(int)

    rows = []
    for year, g in clean.groupby("年"):
        v = g[g["判定"] != "判定不能"]
        n = len(v)
        rows.append({
            "年": year,
            "件数": len(g),
            "判定可能": n,
            "平均スコア": round(v["スコア"].mean(), 4) if n else None,
            "ポジ比率": round((v["判定"] == "ポジティブ").sum() / n, 3) if n else None,
            "ネガ比率": round((v["判定"] == "ネガティブ").sum() / n, 3) if n else None,
        })

    yearly = pd.DataFrame(rows).sort_values("年")
    yearly.to_csv(OUT_YEARLY, sep="\t", index=False, encoding="utf-8-sig")
    print(f"\n保存: {OUT_YEARLY}")
    print("\n=== 年次サマリ ===")
    print(yearly.to_string(index=False))

    # ---- 全国面と地方面の比較 ----
    if "備考" in clean.columns:
        clean["面区分"] = clean["備考"].fillna("").apply(
            lambda x: "全国面" if "全国面" in str(x) else "地方面等")
        print("\n=== 全国面 vs 地方面 ===")
        for kubun, g in clean.groupby("面区分"):
            v = g[g["判定"] != "判定不能"]
            if len(v) == 0:
                continue
            neg = (v["判定"] == "ネガティブ").sum() / len(v)
            pos = (v["判定"] == "ポジティブ").sum() / len(v)
            print(f"  {kubun}: {len(g)}件 / 平均{v['スコア'].mean():+.3f} "
                  f"/ ネガ{neg:.1%} ポジ{pos:.1%}")

    # ---- 年別頻出語 ----
    print("\n=== 年別 頻出語トップ8 ===")
    for year, g in clean.groupby("年"):
        c = Counter()
        for t in g["分かち書き"].fillna(""):
            c.update(str(t).split())
        print(f"{year}: " + ", ".join(f"{w}({n})" for w, n in c.most_common(8)))

    # ---- 確認用 ----
    valid = clean[clean["判定"] != "判定不能"]
    print("\n=== 最もネガティブと判定された5件 ===")
    for _, r in valid.nsmallest(5, "スコア").iterrows():
        print(f"  {r['スコア']:+.2f} [{r['発行日']}] {r['見出し'][:42]}")
        print(f"        {r['寄与語']}")

    print("\n=== 最もポジティブと判定された5件 ===")
    for _, r in valid.nlargest(5, "スコア").iterrows():
        print(f"  {r['スコア']:+.2f} [{r['発行日']}] {r['見出し'][:42]}")
        print(f"        {r['寄与語']}")

    print("\n※ 寄与語を目視で確認し、誤判定の原因語を domain_pn.csv に追加すること。")
    print("※ 発表資料には辞書の出典（小林ら2005 / 東山ら2008）を明記すること。")


if __name__ == "__main__":
    main()