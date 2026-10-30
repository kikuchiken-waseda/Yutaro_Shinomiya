"""5品目の新聞見出しを、一般辞書のみ/自作辞書追加で比較する。
使い方: python3 data_compare_all.py --input 'headlines.tsv'
必要: pip install pandas mecab-python3 ipadic
辞書: wago.121808.pn, pn.csv.m3.120408.trim
自作辞書: domain_pn.csv（コメだけに適用）、domain_momo.csv,
          domain_negi.csv, domain_tomato.csv, domain_ushi.csv（任意）
自作辞書形式: 語,スコア,備考  スコア範囲 -1..1
再構成版: ローカルの data.py と data_compare_all.py を基に機能を統合。
年別集計、面区分比較、年別頻出語、スコア上位下位例をTSV出力。
品目別の年次サマリ・面区分比較・頻出語・ネガポジ5件を端末と report_by_item.txt に出力。
年ごとの推移を yearly_trends.html（グラフ）と yearly_trends.csv に出力。
コメのノイズ候補は集計から除外し、明細には保持。重複候補は自動削除しない。
辞書出典: 小林ら(2005)、東山ら(2008)。
"""
import argparse
import html
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

TARGET_POS = {"名詞", "動詞", "形容詞", "副詞"}
THRESHOLD = 0.10
MIN_POLAR_WORDS = 1
NOISE_MARKERS = ["コメント", "コメリ", "コメディ", "コメて", "コメた"]
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



def export_enhanced(result, output_dir):
    """data.py の確認機能を品目別・辞書条件別に保存する。"""
    def save(frame, name):
        frame.to_csv(output_dir / name, sep="\t", index=False, encoding="utf-8-sig")

    save(result[result["ノイズ"] == "要確認"], "noise_candidates.tsv")
    save(result[result["判定変化"]], "changed_judgments.tsv")
    clean = result[result["ノイズ"] != "要確認"].copy()
    rows = []
    for keys in (["品目", "年"], ["品目", "面区分"], ["品目", "年", "面区分"]):
        rows = []
        for values, g in clean.groupby(keys, sort=True):
            identity = dict(zip(keys, values))
            for mode in ("なし", "あり"):
                col = f"判定_辞書{mode}"
                valid = g[g[col] != "判定不能"]
                n = len(valid)
                counts = valid[col].value_counts()
                rows.append({**identity, "辞書": mode, "件数": len(g),
                    "判定可能": n, "判定不能": len(g)-n,
                    "ポジ件数": int(counts.get("ポジティブ", 0)),
                    "ネガ件数": int(counts.get("ネガティブ", 0)),
                    "中立件数": int(counts.get("中立", 0)),
                    "平均スコア": valid[f"スコア_辞書{mode}"].mean() if n else None,
                    "ポジ比率": counts.get("ポジティブ", 0)/n if n else None,
                    "ネガ比率": counts.get("ネガティブ", 0)/n if n else None,
                    "極性語数中央値": g[f"極性語数_辞書{mode}"].median()})
        filename = {2: "yearly_sentiment.tsv" if "年" in keys else "section_comparison.tsv",
                    3: "yearly_section_comparison.tsv"}[len(keys)]
        save(pd.DataFrame(rows, columns=keys + ["辞書", "件数", "判定可能", "判定不能", "ポジ件数", "ネガ件数", "中立件数", "平均スコア", "ポジ比率", "ネガ比率", "極性語数中央値"]), filename)
    words = []
    examples = []
    for (item, year), g in clean.groupby(["品目", "年"], sort=True):
        counter = Counter(w for text in g["分かち書き"] for w in text.split())
        words.extend({"品目": item, "年": year, "順位": rank, "語": word, "回数": count}
                     for rank, (word, count) in enumerate(counter.most_common(8), 1))
    for item, g in clean.groupby("品目", sort=False):
        for mode in ("なし", "あり"):
            valid = g[g[f"判定_辞書{mode}"] != "判定不能"]
            for label, chosen in (("低スコア", valid.nsmallest(5, f"スコア_辞書{mode}")),
                                  ("高スコア", valid.nlargest(5, f"スコア_辞書{mode}"))):
                for _, row in chosen.iterrows():
                    examples.append({"品目": item, "辞書": mode, "抽出": label,
                        "行ID": row["行ID"], "発行日": row["発行日"], "見出し": row["見出し"],
                        "スコア": row[f"スコア_辞書{mode}"], "判定": row[f"判定_辞書{mode}"],
                        "寄与語": row[f"寄与語_辞書{mode}"]})
    save(pd.DataFrame(words, columns=["品目", "年", "順位", "語", "回数"]), "yearly_top_words.tsv")
    save(pd.DataFrame(examples, columns=["品目", "辞書", "抽出", "行ID", "発行日", "見出し", "スコア", "判定", "寄与語"]), "sentiment_examples.tsv")
    print(f"ノイズ要確認: {(result['ノイズ'] == '要確認').sum()}件（集計から除外、明細には保持）")



def export_yearly_trends(yearly, output_dir):
    """年次集計から、追加ライブラリ不要の推移グラフHTMLを作成する。"""
    yearly.to_csv(output_dir / "yearly_trends.csv", index=False, encoding="utf-8-sig")
    charts = []
    metrics = [("ネガ比率", "ネガティブ比率", 0, 1),
               ("ポジ比率", "ポジティブ比率", 0, 1),
               ("平均スコア", "平均スコア", -1, 1)]
    for item, group in yearly.groupby("品目", sort=False):
        years = sorted(group["年"].astype(int).unique())
        for metric, title, lower, upper in metrics:
            def x(year):
                return 65 + (int(year)-years[0]) / max(years[-1]-years[0], 1) * 620
            def y(value):
                return 235 - (float(value)-lower)/(upper-lower)*200
            svg = ['<svg viewBox="0 0 740 300" role="img" aria-label="' + html.escape(item + title) + '">']
            for i in range(5):
                value = lower + (upper-lower)*i/4
                label = f"{value:.0%}" if lower == 0 else f"{value:+.1f}"
                svg.append(f'<line x1="65" y1="{y(value)}" x2="685" y2="{y(value)}" stroke="#ddd"/><text x="53" y="{y(value)+4}" text-anchor="end">{label}</text>')
            for year in years:
                svg.append(f'<text x="{x(year)}" y="260" text-anchor="middle">{year}</text>')
            for mode, color, dash in (("なし", "#64748b", "6 4"), ("あり", "#2563eb", "")):
                lookup = {int(r["年"]): r for _, r in group[group["辞書"] == mode].iterrows()}
                segment = []
                for year in years:
                    r = lookup.get(year)
                    if r is None or pd.isna(r[metric]):
                        if segment:
                            svg.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
                        segment = []
                        continue
                    px, py = x(year), y(r[metric])
                    segment.append(f"{px},{py}")
                    value = f"{r[metric]:.1%}" if lower == 0 else f"{r[metric]:+.3f}"
                    tip = html.escape(f"{year}年・辞書{mode}: {value} / 判定可能{int(r['判定可能'])}件 / 全{int(r['件数'])}件")
                    svg.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{color}"><title>{tip}</title></circle>')
                if segment:
                    svg.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
            svg.append('</svg>')
            charts.append('<section><h2>' + html.escape(item + '：' + title) + '</h2>' + ''.join(svg) + '</section>')
    table = yearly.to_html(index=False, na_rep="—", border=0, escape=True)
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>品目別・年ごとの推移</title><style>body{font-family:system-ui,sans-serif;margin:32px;color:#172033;background:#f4f6fa}main{max-width:1100px;margin:auto}section{background:white;border-radius:12px;padding:20px;margin:20px 0}svg{width:100%;max-height:400px}svg text{font-size:12px;fill:#475569}table{border-collapse:collapse;background:white;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}.table{overflow:auto}.legend{color:#2563eb}h2{font-size:19px}</style><main><h1>品目別・年ごとの推移</h1><p>ネガ・ポジ比率の分母は「判定可能件数」（中立を含む）。平均スコアも判定可能な見出しで計算。コメのノイズ候補を除外。</p><p><span class="legend">青実線：自作辞書あり</span> ／ 灰破線：自作辞書なし。点にカーソルを合わせると件数と値を表示します。</p><p>判定可能件数が少ない年の比率には注意してください。判定不能のみの年は欠測として線を切り、入力に存在しない年は表示しません。自作辞書未配置の品目では2本の線が重なります。</p>'
    page += ''.join(charts) + '<h2>年次集計表</h2><div class="table">' + table + '</div></main></html>'
    (output_dir / "yearly_trends.html").write_text(page, encoding="utf-8")
    print(f"年ごとの推移グラフ: {output_dir / 'yearly_trends.html'}")



def print_item_reports(result, output_dir):
    """品目ごとの data.py 形式レポートを端末とテキストに出力する。"""
    lines = []
    def emit(text=""):
        lines.append(str(text))

    clean = result[result["ノイズ"] != "要確認"].copy()
    for item in ITEMS:
        g = clean[clean["品目"] == item]
        emit(f"\n{'=' * 20} {item} {'=' * 20}")
        excluded = int(((result["品目"] == item) & (result["ノイズ"] == "要確認")).sum())
        emit(f"集計対象: {len(g)}件 / ノイズ候補除外: {excluded}件")
        if g.empty:
            emit("集計対象の見出しがありません。")
            continue
        for mode in ("なし", "あり"):
            emit(f"\n--- 自作辞書{mode} ---")
            col, scol = f"判定_辞書{mode}", f"スコア_辞書{mode}"
            rows = []
            for year, y in g.groupby("年", sort=True):
                valid = y[y[col] != "判定不能"]
                n = len(valid)
                rows.append({"年": year, "件数": len(y), "判定可能": n,
                    "平均スコア": round(valid[scol].mean(), 4) if n else None,
                    "ポジ比率": round((valid[col] == "ポジティブ").mean(), 3) if n else None,
                    "ネガ比率": round((valid[col] == "ネガティブ").mean(), 3) if n else None})
            emit("\n=== 年次サマリ ===")
            emit(pd.DataFrame(rows).to_string(index=False, na_rep="—"))
            emit("\n=== 全国面 vs 地方面 ===")
            for section, group in g.groupby("面区分", sort=True):
                valid = group[group[col] != "判定不能"]
                n = len(valid)
                if not n:
                    emit(f"  {section}: {len(group)}件 / 判定可能0件 / 平均・比率は算出不可")
                    continue
                neg = (valid[col] == "ネガティブ").mean()
                pos = (valid[col] == "ポジティブ").mean()
                emit(f"  {section}: {len(group)}件 / 平均{valid[scol].mean():+.3f} / ネガ{neg:.1%} ポジ{pos:.1%}")
            emit("\n=== 年別 頻出語トップ8 ===")
            for year, y in g.groupby("年", sort=True):
                counts = Counter(w for text in y["分かち書き"] for w in text.split())
                emit(f"{year}: " + ", ".join(f"{w}({n})" for w, n in counts.most_common(8)))
            for label, order in (("ネガティブ", True), ("ポジティブ", False)):
                emit(f"\n=== 最も{label}と判定された5件 ===")
                candidates = g[g[col] == label].sort_values(scol, ascending=order, kind="stable").head(5)
                if candidates.empty:
                    emit("  該当なし")
                for _, r in candidates.iterrows():
                    emit(f"  {r[scol]:+.2f} [{r['発行日']}] {r['見出し']}")
                    emit(f"        {r[f'寄与語_辞書{mode}']}")
        emit("\n※ 比率の分母は判定可能件数（中立を含む）。頻出語は辞書条件で変わりません。")
    report = "\n".join(lines) + "\n"
    print(report)
    (output_dir / "report_by_item.txt").write_text(report, encoding="utf-8-sig")


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
    if df.empty:
        sys.exit("入力に見出しがありません")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.insert(0, "行ID", range(1, len(df) + 1))
    df["同日同見出し候補"] = pd.Series([duplicate_key(r) for _, r in df.iterrows()]).duplicated(keep=False).to_numpy()
    df[df["同日同見出し候補"]].to_csv(output_dir / "duplicate_candidates.tsv", sep="\t", index=False, encoding="utf-8-sig")

    wago = load_wago(data_dir / "wago.121808.pn")
    meishi = load_meishi(data_dir / "pn.csv.m3.120408.trim")
    if not wago and not meishi:
        sys.exit("既存辞書がありません。--data-dir に辞書のあるフォルダを指定してください")
    print(f"既存辞書: 用言{len(wago)}語 / 名詞{len(meishi)}語")
    import MeCab
    import ipadic
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
            "ノイズ": "要確認" if item == "コメ" and any(w in r["見出し"] for w in NOISE_MARKERS) else "",
            "面区分": "全国面" if "全国面" in str(r["備考"]) else "地方面等",
            "分かち書き": " ".join(b for _, p, b in tokens if p in TARGET_POS and len(b) > 1),
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
    print_item_reports(result, output_dir)
    export_enhanced(result, output_dir)
    # 集計は data.py と同様にノイズ候補を除く。明細には全件を残す。
    result = result[result["ノイズ"] != "要確認"].copy()
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
    yearly = pd.DataFrame(year_rows, columns=["品目", "年", "辞書", "件数", "判定可能", "平均スコア", "ポジ比率", "ネガ比率"])
    yearly.to_csv(output_dir / "yearly_by_item.csv", index=False, encoding="utf-8-sig")
    export_yearly_trends(yearly, output_dir)
    pd.DataFrame([{"品目": i, "語": w, "使用回数": n, "登録スコア": domains[i][w]}
                  for (i, w), n in usage.most_common()], columns=["品目", "語", "使用回数", "登録スコア"]).to_csv(
        output_dir / "domain_usage.csv", index=False, encoding="utf-8-sig")
    print("\n=== 品目別比較 ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n完了: {output_dir}")
    print("※ 辞書が未作成の品目は『自作辞書あり』も既存辞書のみの結果と同じです。")
    print("※ 判定可能数が少ない品目・年の比率は慎重に解釈してください。")


if __name__ == "__main__":
    main()
