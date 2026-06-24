"""Quick quality checks on the raw data before preprocessing."""
import argparse
import glob
import pandas as pd
from loguru import logger


def check_raw(path: str):
    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")
    print(f"Total records:        {len(df)}")
    print(f"Unique PMIDs:         {df['pmid'].nunique()}")
    print(f"Missing abstracts:    {df['abstract'].isna().sum()}")
    print(f"Missing years:        {df['year'].isna().sum()}")
    print(f"Missing titles:       {df['title'].isna().sum()}")

    years = pd.to_numeric(df['year'], errors='coerce')
    year_min = years.min()
    year_max = years.max()
    yr_str = (
        f"{int(year_min)} – {int(year_max)}"
        if pd.notna(year_min) and pd.notna(year_max)
        else "N/A"
    )
    print(f"Year range:           {yr_str}")
    print(f"Median abstract len:  {df['abstract'].dropna().str.len().median():.0f} chars")

    print(f"\nRecords per decade:")
    df['_year_num'] = years
    df['decade'] = (df['_year_num'].dropna() // 10 * 10).astype('Int64')
    print(df.groupby('decade').size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path or glob to raw CSV(s)")
    args = parser.parse_args()
    for path in sorted(glob.glob(args.input)):
        check_raw(path)
