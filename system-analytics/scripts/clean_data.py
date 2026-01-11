"""
clean_data.py
=============
Cleans and preprocesses the raw system log data.

Cleaning Steps:
1. Parse timestamps to datetime
2. Remove invalid response times (negative values)
3. Remove invalid status codes
4. Add derived columns (hour, day_of_week, date)
5. Save cleaned data
"""

import pandas as pd
import numpy as np
import os


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records")
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp column to datetime."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Parsed timestamps: {df['timestamp'].dtype}")
    return df


def remove_invalid_response_times(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records with invalid (negative) response times."""
    df = df.copy()
    invalid_mask = df['response_time_ms'] < 0
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        print(f"Removing {invalid_count} records with invalid response times")
        df = df[~invalid_mask]
    else:
        print("No invalid response times found")
    
    return df


def remove_invalid_status_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records with invalid status codes."""
    df = df.copy()
    
    # Valid HTTP status codes we expect
    valid_codes = [200, 201, 400, 401, 403, 404, 500, 502, 503]
    invalid_mask = ~df['status_code'].isin(valid_codes)
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        print(f"Removing {invalid_count} records with invalid status codes")
        # Show what codes we're removing
        invalid_codes = df.loc[invalid_mask, 'status_code'].unique()
        print(f"  Invalid codes found: {invalid_codes}")
        df = df[~invalid_mask]
    else:
        print("No invalid status codes found")
    
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful derived columns for analysis:
    - hour: hour of day (0-23)
    - day_of_week: day name (Monday-Sunday)
    - day_of_week_num: day number (0=Monday, 6=Sunday)
    - date: date only (no time)
    - is_weekend: boolean for weekend
    - is_peak_hour: boolean for peak traffic hours (6-10 PM)
    """
    df = df.copy()
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['day_of_week_num'] = df['timestamp'].dt.dayofweek
    df['date'] = df['timestamp'].dt.date
    df['is_weekend'] = df['day_of_week_num'] >= 5
    df['is_peak_hour'] = df['hour'].between(18, 22)
    
    print("Added derived columns: hour, day_of_week, day_of_week_num, date, is_weekend, is_peak_hour")
    return df


def ensure_correct_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns have correct data types."""
    df = df.copy()
    
    df['response_time_ms'] = df['response_time_ms'].astype(int)
    df['status_code'] = df['status_code'].astype(int)
    df['error'] = df['error'].astype(bool)
    
    print("Verified data types")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Run validation checks on cleaned data."""
    print("\n--- Data Validation ---")
    
    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("WARNING: Null values found:")
        print(null_counts[null_counts > 0])
    else:
        print("✓ No null values")
    
    # Check response time range
    min_rt = df['response_time_ms'].min()
    max_rt = df['response_time_ms'].max()
    print(f"✓ Response time range: {min_rt}ms - {max_rt}ms")
    
    # Check status codes
    unique_codes = sorted(df['status_code'].unique())
    print(f"✓ Status codes present: {unique_codes}")
    
    # Check date range
    date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
    print(f"✓ Date range: {date_range}")
    
    # Check services
    services = df['service'].unique().tolist()
    print(f"✓ Services: {services}")


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics of cleaned data."""
    print("\n--- Cleaned Data Summary ---")
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique days: {df['date'].nunique()}")
    
    print("\nRequests by service:")
    print(df['service'].value_counts().to_string())
    
    print(f"\nOverall error rate: {df['error'].mean()*100:.2f}%")
    print(f"Mean response time: {df['response_time_ms'].mean():.1f}ms")
    print(f"Median response time: {df['response_time_ms'].median():.1f}ms")
    print(f"P95 response time: {df['response_time_ms'].quantile(0.95):.1f}ms")
    print(f"P99 response time: {df['response_time_ms'].quantile(0.99):.1f}ms")


def main():
    """Main function to clean and save data."""
    # Paths
    script_dir = os.path.dirname(__file__)
    raw_path = os.path.join(script_dir, '..', 'data', 'raw', 'system_logs.csv')
    processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
    processed_path = os.path.join(processed_dir, 'clean_logs.csv')
    
    # Ensure output directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load data
    df = load_raw_data(raw_path)
    initial_count = len(df)
    
    print("\n--- Cleaning Steps ---")
    
    # Clean data
    df = parse_timestamps(df)
    df = remove_invalid_response_times(df)
    df = remove_invalid_status_codes(df)
    df = add_derived_columns(df)
    df = ensure_correct_dtypes(df)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Validate
    validate_data(df)
    
    # Summary
    print_summary_stats(df)
    
    # Calculate records removed
    removed_count = initial_count - len(df)
    print(f"\nRecords removed during cleaning: {removed_count} ({removed_count/initial_count*100:.2f}%)")
    
    # Save cleaned data
    df.to_csv(processed_path, index=False)
    print(f"\nCleaned data saved to: {processed_path}")


if __name__ == '__main__':
    main()
