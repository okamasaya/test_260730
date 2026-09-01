import json
import os
from google import genai
from google.genai import types

# 定数設定
TARGET_FILE = "problems.json"
REPORT_FILE = "report.md"
REPORT_TRIGGER_COUNT = 50  # 何件貯まったらレポートを作成/更新するか

def load_problems():
    if os.path.exists(TARGET_FILE):
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_problems(data):
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_problem(client):
    prompt = """
    以下のフォーマットで、オリジナル試験問題を1問生成してください。
    必ずJSONオブジェクト単体（配列にしない）で出力してください。

    {
      "title": "問題タイトル",
      "question": "問題文",
      "options": ["選択肢1", "選択肢2", "選択肢3", "選択肢4"],
      "answer": "正解の選択肢",
      "explanation": "解説"
    }
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7
        )
    )
    return json.loads(response.text)

def check_duplicates(problems):
    """単純なタイトル・問題文一致チェックの例（必要に応じて類似度判定へ拡張可能）"""
    seen_titles = set()
    duplicates = []
    
    for idx, item in enumerate(problems):
        title = item.get("title", "")
        if title in seen_titles:
            duplicates.append((idx, title))
        else:
            seen_titles.add(title)
            
    return duplicates

def generate_report(problems, duplicates):
    total = len(problems)
    dup_count = len(duplicates)
    unique_count = total - dup_count
    
    report_content = f"""# 問題生成・重複チェックレポート

- **総問題数**: {total} 件
- **ユニーク問題数**: {unique_count} 件
- **検知された重複数**: {dup_count} 件

## 重複検知リスト
"""
    if duplicates:
        for idx, title in duplicates:
            report_content += f"- Index {idx}: `{title}`\n"
    else:
        report_content += "重複は見つかりませんでした。\n"
        
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)
    print("Report generated successfully.")

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    problems = load_problems()

    # 1回の実行で1問追加生成（ペースに応じてループ数を変更可能）
    print("Generating new problem...")
    new_problem = generate_problem(client)
    problems.append(new_problem)
    
    save_problems(problems)
    print(f"Current total problems: {len(problems)}")

    # 指定件数に達しているか確認し重複チェック＆レポート生成
    if len(problems) >= REPORT_TRIGGER_COUNT and len(problems) % 10 == 0:
        print("Running duplicate check and updating report...")
        duplicates = check_duplicates(problems)
        generate_report(problems, duplicates)

if __name__ == "__main__":
    main()
