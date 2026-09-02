import json
import os
import time
from datetime import datetime, timedelta
from google import genai
from google.genai import types

# 定数設定
TARGET_FILE = "problems.json"
REPORT_FILE = "report.md"
REPORT_TRIGGER_COUNT = 50
INTERVAL_HOURS = 6  # 実行間隔（時間単位で指定）

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
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7
        )
    )
    
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
        
    return json.loads(text.strip())

def check_duplicates(problems):
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

def run_task(client):
    """1回分の生成・チェック処理"""
    problems = load_problems()

    print("Generating new problem...")
    new_problem = generate_problem(client)
    problems.append(new_problem)
    
    save_problems(problems)
    print(f"Current total problems: {len(problems)}")

    if len(problems) >= REPORT_TRIGGER_COUNT and len(problems) % 10 == 0:
        print("Running duplicate check and updating report...")
        duplicates = check_duplicates(problems)
        generate_report(problems, duplicates)

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    interval_seconds = INTERVAL_HOURS * 3600

    print("=== 自動生成プログラムを開始しました (終了するには Ctrl+C) ===")
    
    while True:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now_str}] 処理を実行します...")
        
        try:
            run_task(client)
        except Exception as e:
            # 通信エラーや一時的なAPIエラーが起きてもプログラムを殺さずにスキップする
            print(f"エラーが発生しました（次回実行時にリトライします）: {e}")
        
        next_run = datetime.now() + timedelta(seconds=interval_seconds)
        print(f"次回実行予定: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    main()
