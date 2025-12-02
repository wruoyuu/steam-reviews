"""
Merge all individual review JSON files into a single all_reviews.csv
Run this script to combine your collected data without re-downloading
"""

import pandas as pd
import json
from pathlib import Path

def merge_review_files(input_dir='data/raw', output_file='data/raw/all_reviews.csv'):
    """
    Merge all individual game review JSON files into one CSV
    """
    input_path = Path(input_dir)
    
    # Find all JSON files
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        print(f"❌ No JSON files found in {input_dir}")
        return None
    
    print(f"Found {len(json_files)} game review files")
    print(f"Merging reviews...\n")
    
    all_reviews = []
    
    for json_file in json_files:
        try:
            # Load JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
            
            # Add to combined list
            all_reviews.extend(reviews)
            
            # Show progress
            game_name = reviews[0]['game_name'] if reviews else 'Unknown'
            print(f"  ✓ {game_name}: {len(reviews)} reviews")
            
        except Exception as e:
            print(f"  ⚠ Error reading {json_file.name}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_reviews)
    
    # Save to CSV
    output_path = Path(output_file)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully merged {len(all_reviews)} reviews from {len(json_files)} games")
    print(f"✓ Saved to: {output_path}")
    print(f"{'='*60}")
    
    # Show summary statistics
    print(f"\nDataset Summary:")
    print(f"  Total reviews: {len(df)}")
    print(f"  Unique games: {df['game_name'].nunique()}")
    print(f"  Date range: {pd.to_datetime(df['timestamp_created'], unit='s').min()} to {pd.to_datetime(df['timestamp_created'], unit='s').max()}")
    print(f"\nTop 5 games by review count:")
    print(df['game_name'].value_counts().head())
    
    return df

if __name__ == "__main__":
    print("="*60)
    print("Steam Review Data Merger")
    print("="*60 + "\n")
    
    # Merge the files
    df = merge_review_files()
    
    if df is not None:
        print("\n✓ All done! You can now run preprocessing.py")
    else:
        print("\n❌ Failed to merge files. Check that data/raw/ contains JSON files.")