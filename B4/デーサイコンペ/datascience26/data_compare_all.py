"""5品目の新聞見出しを、一般辞書のみ/自作辞書追加で比較する。
使い方: python3 data_compare_all.py --input 'headlines.tsv'
必要: pip install pandas mecab-python3 ipadic
辞書: wago.121808.pn, pn.csv.m3.120408.trim
自作辞書: domain_pn.csv（コメだけに適用）、domain_momo.csv,
          domain_negi.csv, domain_tomato.csv, domain_ushi.csv（任意）
自作辞書形式: 語,スコア,備考  スコア範囲 -1..1
"""
import argparse
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import MeCab
import ipadic

TARGET_POS = {"名詞", "動詞", "形容詞", "副詞"}
THRESHOLD = 0.10
MIN_POLAR_WORDS = 1
ITEMS = ["コメ", "モモ", "ネギ", "トマト", "牛"]
DOMAIN_FILES = {
    "コメ": "domain_pn.csv", "モモ": "domain_momo.csv",
    "ネギ": "domain_negi.csv", "トマト": "domain_tomato.csv", "牛": "domain_ushi.csv",
}


def read_lines(path):
    for enc in ("utf-8-sig", "utf-8", "euc_jp", "shift_jis", "cp932"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"辞書の文字コードを読み取れません: {path}")


def load_wago(path):
    table = {}
    if not path.is_file():
        return table
    for line in read_lines(path):
        parts = [v.strip() for v in line.split("\t") if v.strip()]
        if len(parts) < 2:
            continue
        a, b = parts[:2]
        if a.startswith(("ポジ", "ネガ")):
            label, word = a, b
        elif b.startswith(("ポジ", "ネガ")):
            label, word = b, a
        else:
            continue
        word = word.replace(" ", "").replace("　", "")
        if word:
            table[word] = 1.0 if label.startswith("ポジ") else -1.0
    return table


def load_meishi(path):
    table = {}
    if not path.is_file():
        return table
    values = {"p": 1.0, "n": -1.0, "e": 0.0}
    for line in read_lines(path):
        parts = [v.strip() for v in line.split("\t") if v.strip()]
        if len(parts) < 2:
            continue
        a, b = parts[:2]
        if b in values:
            word, label = a, b
        elif a in values:
            word, label = b, a
        else:
            continue
        word = word.replace(" ", "").replace("　", "")
        if word:
            table[word] = values[label]
    return table


def load_domain(path):
    if not path.is_file():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"語": str})
    if not {"語", "スコア"}.issubset(df.columns):
        raise ValueError(f"{path.name} に「語」「スコア」列がありません")
    result = {}
    for _, r in df.iterrows():
        word = str(r["語"]).strip()
        score = float(r["スコア"])
        if not word or not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError(f"{path.name}: 不正な語・スコア: {word} / {score}")
        result[word] = score
    return result


def tokenize(tagger, text):
    node = tagger.parseToNode(str(text))
    items = []
    while node:
        if node.surface:
            fs = node.feature.split(",")
            base = fs[6] if len(fs) > 6 and fs[6] != "*" else node.surface
            items.append((node.surface, fs[0], base))
        node = node.next
    return items


def score(tokens, wago, meishi, domain):
    hits, details, domain_words = [], [], []
    for surface, pos, base in tokens:
        if pos not in TARGET_POS:
            continue
        for key in (base, surface):
            if key in domain:
                v, origin = domain[key], "補助"
                domain_words.append(key)
            elif key in wago:
                v, origin = wago[key], "用言"
            elif key in meishi:
                v, origin = meishi[key], "名詞"
            else:
                continue
            hits.append(v)
            details.append(f"{key}:{v:+.2f}({origin})")
            break
    s = sum(hits) / len(hits) if hits else 0.0
    polar = sum(v != 0 for v in hits)
    label = ("判定不能" if polar < MIN_POLAR_WORDS else
             "ポジティブ" if s > THRESHOLD else
             "ネガティブ" if s < -THRESHOLD else "中立")
    return s, len(hits), polar, label, " ".join(details), domain_words


def duplicate_key(row):
    # 表記ゆれ・空白が異なる同日・同品目の見出しを確認用に抽出する（自動削除しない）
    title = unicodedata.normalize("NFKC", str(row["見出し"]))
    title = re.sub(r"\s+", "", title).replace("／", "/")
    return (row["品目"], row["発行日"], title)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=root / "headlines.tsv")
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "results_all")
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        sys.exit(f"入力がありません: {input_path}")
    df = pd.read_csv(input_path, sep="\t", dtype=str, encoding="utf-8-sig")
    required = ["品目", "紙名", "発行日", "年", "見出し", "備考"]
    missing = set(required) - set(df.columns)
    if missing:
        sys.exit(f"列が不足しています: {sorted(missing)}")
    if df[["品目", "発行日", "年", "見出し"]].isna().any().any():
        sys.exit("品目・発行日・年・見出しに欠損があります")
    unknown = sorted(set(df["品目"]) - set(ITEMS))
    if unknown:
        sys.exit(f"想定外の品目: {unknown}")
    if (df["発行日"].str[:4] != df["年"]).any():
        sys.exit("発行日と年が一致しない行があります")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.insert(0, "行ID", range(1, len(df) + 1))
    df["同日同見出し候補"] = pd.Series([duplicate_key(r) for _, r in df.iterrows()]).duplicated(keep=False).to_numpy()
    df[df["同日同見出し候補"]].to_csv(output_dir / "duplicate_candidates.tsv", sep="\t", index=False, encoding="utf-8-sig")

    wago = load_wago(data_dir / "wago.121808.pn")
    meishi = load_meishi(data_dir / "pn.csv.m3.120408.trim")
    if not wago and not meishi:
        sys.exit("既存辞書がありません。--data-dir に辞書のあるフォルダを指定してください")
    print(f"既存辞書: 用言{len(wago)}語 / 名詞{len(meishi)}語")
    tagger = MeCab.Tagger(ipadic.MECAB_ARGS)
    records = []
    usage = Counter()
    # 各品目の辞書は別々に読み、コメ用の辞書を他品目に流用しない。
    domains = {item: load_domain(data_dir / DOMAIN_FILES[item]) for item in ITEMS}
    for item in ITEMS:
        print(f"{item}: {len(df[df['品目'] == item])}件 / 自作辞書{len(domains[item])}語 ({DOMAIN_FILES[item]})")
    for _, r in df.iterrows():
        item = r["品目"]
        tokens = tokenize(tagger, r["見出し"])
        no = score(tokens, wago, meishi, {})
        yes = score(tokens, wago, meishi, domains[item])
        usage.update((item, w) for w in yes[5])
        records.append({
            **r.to_dict(),
            "スコア_辞書なし": round(no[0], 4), "判定_辞書なし": no[3],
            "ヒット数_辞書なし": no[1], "極性語数_辞書なし": no[2], "寄与語_辞書なし": no[4],
            "スコア_辞書あり": round(yes[0], 4), "判定_辞書あり": yes[3],
            "ヒット数_辞書あり": yes[1], "極性語数_辞書あり": yes[2], "寄与語_辞書あり": yes[4],
            "自作辞書使用": bool(yes[5]), "自作辞書使用語": " ".join(yes[5]),
            "スコア差": yes[0] - no[0],
            "スコア変化": not math.isclose(yes[0], no[0], abs_tol=1e-12, rel_tol=0),
            "判定変化": yes[3] != no[3],
        })
    result = pd.DataFrame(records)
    result.to_csv(output_dir / "headlines_all_comparison.tsv", sep="\t", index=False, encoding="utf-8-sig")
    result.to_csv(output_dir / "headlines_all_comparison.csv", index=False, encoding="utf-8-sig")
    year_rows = []
    summary_rows = []
    for item, g in result.groupby("品目", sort=False):
        row = {"品目": item, "件数": len(g), "自作辞書使用件数": int(g["自作辞書使用"].sum()),
               "スコア変化件数": int(g["スコア変化"].sum()), "判定変化件数": int(g["判定変化"].sum())}
        for mode in ("なし", "あり"):
            valid = g[g[f"判定_辞書{mode}"] != "判定不能"]
            row[f"判定可能件数_辞書{mode}"] = len(valid)
            row[f"ネガ件数_辞書{mode}"] = int((valid[f"判定_辞書{mode}"] == "ネガティブ").sum())
            row[f"ポジ件数_辞書{mode}"] = int((valid[f"判定_辞書{mode}"] == "ポジティブ").sum())
            row[f"ネガ比率_辞書{mode}"] = round(row[f"ネガ件数_辞書{mode}"] / len(valid), 4) if len(valid) else None
        summary_rows.append(row)
        for year, y in g.groupby("年", sort=True):
            for mode in ("なし", "あり"):
                valid = y[y[f"判定_辞書{mode}"] != "判定不能"]
                col = f"判定_辞書{mode}"
                year_rows.append({"品目": item, "年": year, "辞書": mode, "件数": len(y),
                                  "判定可能": len(valid),
                                  "平均スコア": round(valid[f"スコア_辞書{mode}"].mean(), 4) if len(valid) else None,
                                  "ポジ比率": round((valid[col] == "ポジティブ").mean(), 4) if len(valid) else None,
                                  "ネガ比率": round((valid[col] == "ネガティブ").mean(), 4) if len(valid) else None})
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_by_item.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(year_rows).to_csv(output_dir / "yearly_by_item.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"品目": i, "語": w, "使用回数": n, "登録スコア": domains[i][w]}
                  for (i, w), n in usage.most_common()]).to_csv(
        output_dir / "domain_usage.csv", index=False, encoding="utf-8-sig")
    print("\n=== 品目別比較 ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n完了: {output_dir}")
    print("※ 辞書が未作成の品目は『自作辞書あり』も既存辞書のみの結果と同じです。")
    print("※ 判定可能数が少ない品目・年の比率は慎重に解釈してください。")


if __name__ == "__main__":
    main()
