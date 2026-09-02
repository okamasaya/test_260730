"""
JLPT 問題自動生成スクリプト（API最適化版）
対応区分: N1/N2 言語知識（文字・語彙・文法）

主な変更点:
  - 1問/呼び出し → 10問バッチ生成（API呼び出しコスト 1/10）
  - system_instruction で固定ルールを分離
  - response_schema（Pydantic）で型安全なJSON出力
  - 既出語リストをファイル永続化し、バッチ間で引き継ぎ
  - セッションリフレッシュ（Nバッチごとに再構築）
  - 指数バックオフ付きリトライ
  - 6時間インターバル → 連続バッチ＋短インターバルに変更
  - 重複検知をタイトル一致 → used_word ベースに変更
"""

import json
import os
import time
from datetime import datetime
from pydantic import BaseModel
from google import genai
from google.genai import types

# ============================================================
# 定数
# ============================================================
TARGET_FILE        = "problems.json"
USED_WORDS_FILE    = "used_words.json"
REPORT_FILE        = "report.md"

BATCH_SIZE           = 10   # 1回のAPI呼び出しで生成する問題数
SESSION_REFRESH_EVERY = 5   # Nバッチごとに既出語リストを再構築
BATCH_INTERVAL_SEC   = 3    # バッチ間スリープ秒数（レート制限対策）
RETRY_MAX            = 3    # リトライ上限
RETRY_WAIT_BASE      = 5    # 指数バックオフの基底秒数（5→10→20秒）

# ============================================================
# 実行パラメータ（バッチ実行ごとに変更する箇所）
# ============================================================
DIFFICULTY    = 3                # 1〜5
CATEGORY_LV3  = "漢字読み"       # 文字: 漢字読み
                                 # 語彙: 文脈規定 / 言い換え類義 / 用法
                                 # 文法: 文の文法1 / 文の文法2 / 文章の文法
EXAM_PREFIX   = "JLPT_N1_LK_Ch" # question_id のプレフィックス
TOTAL_BATCHES = 10               # 今回の実行で生成するバッチ数

# 使用するモデル名（要確認：Gemini API コンソールで有効なモデル名を設定）
MODEL_NAME = "gemini-2.5-flash"

# ============================================================
# レスポンススキーマ（Pydantic）
# response_schema に渡すことで、JSON の型・構造を API 側で保証する
# ============================================================
class ProblemItem(BaseModel):
    question_id  : str       # 例: JLPT_N1_LK_Ch_001
    question     : str       # 問題文
    options      : list[str] # 選択肢4つ（ひらがな表記）
    answer       : str       # 正解（選択肢の文字列と一致させる）
    explanation  : str       # Phase-3 で作成 → 固定値「（Phase-3で作成）」
    difficulty   : int       # 1〜5
    category_lv3 : str       # 漢字読み / 文脈規定 / etc.
    used_word    : str       # バッチで使用した出題語（重複管理用キー）

# ============================================================
# システム命令（固定ルール）
# ここに実際の prob_gen_JLPT_N1_LK_Ch_v2.md の「固定部分」を貼り付ける。
# 制御パラメータ（difficulty・既出語リスト等）は user_prompt 側に渡すこと。
# ============================================================
SYSTEM_INSTRUCTION = """
あなたはJLPT N1言語知識（文字）の模擬問題作成の専門家です。
以下のルールを厳守してください。

【出題形式 - 漢字読み】
- 問題文中の下線部（出題語）の読み方を4択で問う形式
- 選択肢はすべてひらがな表記
- 正解1つ、誤答3つ（音訓の混同・部首読みなど紛らわしい読みを使用）

【品質基準】
- 問題文のレベル・場面はN1相当（ビジネス/報道/公文書/日常/文化系）
- ルビは一切付けない
- explanation には必ず固定値「（Phase-3で作成）」を入れること
- 既出語リストに含まれる語を出題語に使用しないこと
- バッチ内で同じ出題語・類似問題文を重複させないこと

【場面制約（10問中の配分）】
- 最低3カテゴリ以上の場面を使用
  （業務系 / 生活基盤系 / 社会系 / 日常系 / 文化系）
- 最低1問は「報道 / 災害・防災 / 救急医療 / 警察・治安 / 法律・裁判」
  のいずれかの場面を含めること

【出力形式】
- JSON配列のみを出力すること
- 説明文・マークダウン記法（```など）は一切出力しないこと
""".strip()

# ============================================================
# データ管理
# ============================================================
def load_problems() -> list:
    if os.path.exists(TARGET_FILE):
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_problems(data: list):
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_used_words() -> list[str]:
    if os.path.exists(USED_WORDS_FILE):
        with open(USED_WORDS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_used_words(words: list[str]):
    with open(USED_WORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

def extract_used_words(problems: list) -> list[str]:
    """problems.json から既出語リストを再構築（セッションリフレッシュ用）"""
    seen = {}
    for p in problems:
        w = p.get("used_word", "").strip()
        if w:
            seen[w] = True
    return list(seen.keys())

# ============================================================
# プロンプト構築
# 制御パラメータ（可変部分）を user_prompt として組み立てる
# ============================================================
def build_user_prompt(start_id: int, difficulty: int,
                      category_lv3: str, used_words: list[str]) -> str:
    # 直近200語のみ渡す（プロンプト肥大化防止）
    recent_words = used_words[-200:] if len(used_words) > 200 else used_words
    used_words_str = "、".join(recent_words) if recent_words else "なし"

    return f"""以下の制御パラメータで {BATCH_SIZE} 問を生成してください。

【制御パラメータ】
- difficulty: {difficulty}（1=最易〜5=最難）
- category_lv3: {category_lv3}
- 開始ID: {EXAM_PREFIX}_{start_id:03d}（以降連番で採番）
- バッチサイズ: {BATCH_SIZE} 問

【既出語リスト（出題語に使用禁止）】
{used_words_str}
""".strip()

# ============================================================
# API呼び出し（指数バックオフ付きリトライ）
# ============================================================
def generate_batch(client: genai.Client, start_id: int, difficulty: int,
                   category_lv3: str, used_words: list[str]) -> list[dict]:
    user_prompt = build_user_prompt(start_id, difficulty, category_lv3, used_words)

    for attempt in range(1, RETRY_MAX + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=list[ProblemItem],  # 型安全なJSON出力
                    temperature=0.9,                    # 多様性確保（0.8〜1.0推奨）
                    max_output_tokens=4096,
                )
            )
            batch = json.loads(response.text)
            if not isinstance(batch, list):
                batch = [batch]  # 万一オブジェクトで返ってきた場合の保険
            return batch

        except Exception as e:
            wait = RETRY_WAIT_BASE * (2 ** (attempt - 1))  # 5 → 10 → 20秒
            print(f"  [attempt {attempt}/{RETRY_MAX}] エラー: {e}")
            if attempt < RETRY_MAX:
                print(f"  → {wait}秒後にリトライ...")
                time.sleep(wait)

    raise RuntimeError(f"バッチ生成失敗（{RETRY_MAX}回リトライ済み）")

# ============================================================
# 重複チェック（used_word ベース）
# ============================================================
def check_duplicates(problems: list) -> list[tuple]:
    """(現在index, 出題語, 初出index) のリストを返す"""
    seen: dict[str, int] = {}
    duplicates = []
    for idx, item in enumerate(problems):
        word = item.get("used_word", "")
        if word in seen:
            duplicates.append((idx, word, seen[word]))
        else:
            seen[word] = idx
    return duplicates

# ============================================================
# レポート生成
# ============================================================
def generate_report(problems: list):
    duplicates = check_duplicates(problems)
    total     = len(problems)
    dup_count = len(duplicates)

    lines = [
        f"# 問題生成レポート",
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- 総問題数: {total} 件",
        f"- ユニーク問題数: {total - dup_count} 件",
        f"- used_word 重複: {dup_count} 件",
        "",
        "## 重複リスト",
    ]
    if duplicates:
        for idx, word, first_idx in duplicates:
            lines.append(f"- Index {idx}: `{word}`（初出: Index {first_idx}）")
    else:
        lines.append("重複は検出されませんでした。")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"レポート生成: {REPORT_FILE}")

# ============================================================
# メインループ（連続バッチ生成）
# ============================================================
def run_batch_loop(client: genai.Client):
    problems   = load_problems()
    used_words = extract_used_words(problems)
    start_id   = len(problems) + 1

    print(f"既存問題数: {len(problems)} 問 / 既出語: {len(used_words)} 語")

    for batch_num in range(1, TOTAL_BATCHES + 1):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] Batch {batch_num}/{TOTAL_BATCHES}  start_id={start_id}")

        # セッションリフレッシュ（Nバッチごとに problems.json から再構築）
        if batch_num > 1 and (batch_num - 1) % SESSION_REFRESH_EVERY == 0:
            print("  → セッションリフレッシュ（既出語リストを再構築）")
            used_words = extract_used_words(problems)

        try:
            batch = generate_batch(client, start_id, DIFFICULTY, CATEGORY_LV3, used_words)
        except RuntimeError as e:
            print(f"  スキップ: {e}")
            continue

        # 既出語を更新して永続化
        new_words = [p.get("used_word", "").strip() for p in batch if p.get("used_word")]
        for w in new_words:
            if w and w not in used_words:
                used_words.append(w)
        save_used_words(used_words)

        # 問題を追記・保存
        problems.extend(batch)
        save_problems(problems)
        start_id += len(batch)

        print(f"  → {len(batch)}問追加  累計: {len(problems)}問 / 既出語: {len(used_words)}語")

        if batch_num < TOTAL_BATCHES:
            time.sleep(BATCH_INTERVAL_SEC)

    generate_report(problems)
    print(f"\n=== 完了: {TOTAL_BATCHES}バッチ / {len(problems)}問 ===")

# ============================================================
# エントリーポイント
# ============================================================
def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY が設定されていません。")

    client = genai.Client(api_key=api_key)

    print(f"=== JLPT問題生成 | {EXAM_PREFIX} | difficulty={DIFFICULTY} | {TOTAL_BATCHES}バッチ ===")
    run_batch_loop(client)

if __name__ == "__main__":
    main()
