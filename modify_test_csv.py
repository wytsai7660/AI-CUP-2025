#!/usr/bin/env python3
import pandas as pd
import sys

def main(input_csv: str, output_csv: str):
    # 讀取 CSV
    df = pd.read_csv(input_csv)
    
    # 將 'hold racket handed' 欄位的值全部改為 0.5
    df['hold racket handed'] = 0.5
    
    # 存回新的 CSV
    df.to_csv(output_csv, index=False)
    print(f"Updated '{output_csv}' with 'hold racket handed' = 0.5 for all rows.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python set_hold_racket.py <input.csv> <output.csv>")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    main(input_csv, output_csv)
