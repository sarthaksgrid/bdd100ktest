import pandas as pd
from pathlib import Path

def get_total_epochs():
    # 1. Path Setup
    SCRIPT_DIR = Path(__file__).resolve().parent
    runs_dir = SCRIPT_DIR / "runs"
    
    # Find the most recent run folder
    all_runs = sorted(list(runs_dir.glob("bdd100k_run*")), reverse=True)
    
    if not all_runs:
        print("❌ No training runs found.")
        return 0

    latest_run = all_runs[0]
    results_csv = latest_run / "results.csv"
    
    if not results_csv.exists():
        print(f"⚠️  No results.csv found in {latest_run.name}. The run might have crashed before saving.")
        return 0

    # 2. Read the CSV
    try:
        df = pd.read_csv(results_csv)
        # Strip whitespace from column names just in case
        df.columns = df.columns.str.strip()
        
        # The 'epoch' column or simply the length of the dataframe tells us progress
        total_completed = len(df)
        
        print(f"\n{'='*40}")
        print(f" 📂 Run Folder: {latest_run.name}")
        print(f" ✅ Total Epochs Completed: {total_completed}")
        print(f" 🚀 Next Epoch Should Be: {total_completed + 1}")
        print(f"{'='*40}\n")
        
        return total_completed
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return 0

if __name__ == "__main__":
    get_total_epochs()