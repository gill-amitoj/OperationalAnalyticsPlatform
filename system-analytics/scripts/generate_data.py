"""
generate_data.py
================
Generates synthetic system/application log data for analytics.

Data Generation Assumptions:
- 10 days of log data (Jan 1-10, 2026)
- Peak traffic in evenings (6-10 PM)
- Different services have different traffic patterns
- Higher traffic periods correlate with higher latency and error rates
- Endpoints have varying performance characteristics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set seed for reproducibility
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date range: 10 days of data
START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 1, 10)
TOTAL_DAYS = (END_DATE - START_DATE).days + 1

# Services and their endpoints
SERVICE_ENDPOINTS = {
    'auth': ['/login', '/register', '/logout', '/reset-password'],
    'events': ['/list', '/create', '/update', '/delete'],
    'payments': ['/checkout', '/refund', '/history', '/verify']
}

# Base requests per hour by service (off-peak)
BASE_REQUESTS_PER_HOUR = {
    'auth': 150,
    'events': 100,
    'payments': 50
}

# Endpoint weights within each service (traffic distribution)
ENDPOINT_WEIGHTS = {
    'auth': {'/login': 0.6, '/register': 0.15, '/logout': 0.2, '/reset-password': 0.05},
    'events': {'/list': 0.5, '/create': 0.25, '/update': 0.15, '/delete': 0.1},
    'payments': {'/checkout': 0.4, '/refund': 0.1, '/history': 0.35, '/verify': 0.15}
}

# Base response times (ms) by endpoint - some endpoints are naturally slower
BASE_RESPONSE_TIMES = {
    '/login': 80,
    '/register': 150,
    '/logout': 30,
    '/reset-password': 200,
    '/list': 100,
    '/create': 120,
    '/update': 90,
    '/delete': 70,
    '/checkout': 300,
    '/refund': 250,
    '/history': 150,
    '/verify': 180
}

# Base error rates by endpoint
BASE_ERROR_RATES = {
    '/login': 0.03,
    '/register': 0.05,
    '/logout': 0.01,
    '/reset-password': 0.08,
    '/list': 0.02,
    '/create': 0.04,
    '/update': 0.03,
    '/delete': 0.02,
    '/checkout': 0.06,
    '/refund': 0.07,
    '/history': 0.02,
    '/verify': 0.04
}


def get_traffic_multiplier(hour: int) -> float:
    """
    Returns traffic multiplier based on hour of day.
    Peak hours: 6-10 PM (18-22) → higher traffic
    Low hours: 2-6 AM → lower traffic
    """
    if 18 <= hour <= 22:  # Evening peak
        return 2.5
    elif 12 <= hour <= 14:  # Lunch peak
        return 1.5
    elif 9 <= hour <= 17:  # Business hours
        return 1.2
    elif 2 <= hour <= 6:  # Night low
        return 0.3
    else:  # Other times
        return 0.8


def get_weekend_multiplier(day_of_week: int) -> float:
    """
    Returns traffic multiplier for day of week.
    0 = Monday, 6 = Sunday
    Weekends have different traffic patterns.
    """
    if day_of_week >= 5:  # Weekend
        return 0.7
    elif day_of_week == 0:  # Monday
        return 1.1
    elif day_of_week == 4:  # Friday
        return 1.2
    else:
        return 1.0


def calculate_response_time(endpoint: str, traffic_multiplier: float) -> int:
    """
    Calculate response time with realistic variation.
    Higher traffic → higher latency due to system load.
    """
    base_time = BASE_RESPONSE_TIMES[endpoint]
    
    # Traffic impact: more traffic = slower responses
    traffic_impact = 1 + (traffic_multiplier - 1) * 0.4
    
    # Add random variation (log-normal distribution for realistic tail)
    mean_time = base_time * traffic_impact
    response_time = np.random.lognormal(
        mean=np.log(mean_time),
        sigma=0.3
    )
    
    return max(10, int(response_time))  # Minimum 10ms


def calculate_error(endpoint: str, traffic_multiplier: float, response_time: int) -> tuple:
    """
    Determine if request results in error.
    Higher traffic and slower responses correlate with more errors.
    Returns (is_error, status_code)
    """
    base_rate = BASE_ERROR_RATES[endpoint]
    
    # Traffic impact: more traffic = more errors
    traffic_impact = 1 + (traffic_multiplier - 1) * 0.5
    
    # Slow response impact: very slow responses more likely to error
    latency_factor = 1.0
    if response_time > BASE_RESPONSE_TIMES[endpoint] * 3:
        latency_factor = 1.5
    elif response_time > BASE_RESPONSE_TIMES[endpoint] * 2:
        latency_factor = 1.2
    
    error_rate = base_rate * traffic_impact * latency_factor
    error_rate = min(error_rate, 0.25)  # Cap at 25%
    
    is_error = np.random.random() < error_rate
    
    if is_error:
        # Distribute error codes realistically
        error_codes = [400, 401, 403, 404, 500, 502, 503]
        weights = [0.25, 0.15, 0.1, 0.15, 0.2, 0.05, 0.1]
        status_code = np.random.choice(error_codes, p=weights)
    else:
        status_code = 200
    
    return is_error, status_code


def generate_logs() -> pd.DataFrame:
    """
    Generate synthetic log data for the specified date range.
    """
    print("Starting data generation...")
    
    logs = []
    current_date = START_DATE
    
    while current_date <= END_DATE:
        day_of_week = current_date.weekday()
        weekend_mult = get_weekend_multiplier(day_of_week)
        
        # Generate logs for each hour of the day
        for hour in range(24):
            traffic_mult = get_traffic_multiplier(hour)
            combined_mult = traffic_mult * weekend_mult
            
            # Generate requests for each service
            for service, endpoints in SERVICE_ENDPOINTS.items():
                # Calculate number of requests for this hour
                base_requests = BASE_REQUESTS_PER_HOUR[service]
                num_requests = int(base_requests * combined_mult)
                
                # Add some randomness to request count
                num_requests = max(1, int(np.random.normal(num_requests, num_requests * 0.1)))
                
                for _ in range(num_requests):
                    # Select endpoint based on weights
                    endpoint_list = list(ENDPOINT_WEIGHTS[service].keys())
                    weights = list(ENDPOINT_WEIGHTS[service].values())
                    endpoint = np.random.choice(endpoint_list, p=weights)
                    
                    # Generate timestamp within the hour
                    minute = np.random.randint(0, 60)
                    second = np.random.randint(0, 60)
                    timestamp = current_date.replace(
                        hour=hour, minute=minute, second=second
                    )
                    
                    # Calculate metrics
                    response_time = calculate_response_time(endpoint, combined_mult)
                    is_error, status_code = calculate_error(
                        endpoint, combined_mult, response_time
                    )
                    
                    logs.append({
                        'timestamp': timestamp,
                        'service': service,
                        'endpoint': endpoint,
                        'response_time_ms': response_time,
                        'status_code': status_code,
                        'error': is_error
                    })
        
        print(f"  Generated data for {current_date.strftime('%Y-%m-%d')}")
        current_date += timedelta(days=1)
    
    # Create DataFrame and sort by timestamp
    df = pd.DataFrame(logs)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nTotal records generated: {len(df):,}")
    return df


def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject some realistic anomalies into the data:
    - A brief outage period
    - Some missing/invalid values
    """
    df = df.copy()
    
    # Simulate a brief service degradation on Jan 5th, 2-3 PM
    degradation_mask = (
        (df['timestamp'].dt.day == 5) & 
        (df['timestamp'].dt.hour >= 14) & 
        (df['timestamp'].dt.hour < 15)
    )
    
    # Increase response times and error rates during degradation
    df.loc[degradation_mask, 'response_time_ms'] = (
        df.loc[degradation_mask, 'response_time_ms'] * 3
    ).astype(int)
    
    # Set some requests to 503 during degradation
    degraded_indices = df[degradation_mask].sample(frac=0.3).index
    df.loc[degraded_indices, 'status_code'] = 503
    df.loc[degraded_indices, 'error'] = True
    
    # Inject a few invalid/missing values (for cleaning exercise)
    # Set ~0.1% of response times to invalid values
    invalid_count = max(1, int(len(df) * 0.001))
    invalid_indices = df.sample(n=invalid_count).index
    df.loc[invalid_indices, 'response_time_ms'] = -1
    
    # Set a few status codes to unusual values
    unusual_count = max(1, int(len(df) * 0.0005))
    unusual_indices = df.sample(n=unusual_count).index
    df.loc[unusual_indices, 'status_code'] = 999
    
    print(f"Injected anomalies: {len(degraded_indices)} degraded requests, "
          f"{invalid_count} invalid response times, {unusual_count} unusual status codes")
    
    return df


def main():
    """Main function to generate and save log data."""
    # Generate logs
    df = generate_logs()
    
    # Inject some anomalies for realistic data cleaning exercise
    df = inject_anomalies(df)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'system_logs.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\nData saved to: {output_path}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Services: {df['service'].unique().tolist()}")
    print(f"Total requests: {len(df):,}")
    
    # Quick summary stats
    print("\n--- Quick Summary ---")
    print(f"Error rate: {df['error'].mean()*100:.2f}%")
    print(f"Avg response time: {df['response_time_ms'].mean():.1f}ms")
    print(f"P95 response time: {df['response_time_ms'].quantile(0.95):.1f}ms")


if __name__ == '__main__':
    main()
