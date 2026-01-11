"""
analysis.py
===========
Performs analytics on cleaned system log data and generates visualizations.

Metrics Computed:
1. Requests per hour (traffic patterns)
2. Average and P95 response time (performance)
3. Error rate over time (reliability)
4. Performance by endpoint (bottleneck identification)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_clean_data(filepath: str) -> pd.DataFrame:
    """Load cleaned CSV data and parse timestamps."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df):,} records")
    return df


def save_figure(fig, filename: str, viz_dir: str) -> None:
    """Save figure to visualizations directory."""
    filepath = os.path.join(viz_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {filename}")


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_requests_per_hour(df: pd.DataFrame, viz_dir: str) -> pd.DataFrame:
    """
    Analyze and visualize requests per hour.
    Shows traffic patterns throughout the day.
    """
    print("\n--- Requests Per Hour Analysis ---")
    
    # Calculate requests per hour
    hourly_requests = df.groupby('hour').size().reset_index(name='requests')
    
    # Calculate averages for peak vs off-peak
    peak_avg = df[df['is_peak_hour']].groupby('hour').size().mean()
    offpeak_avg = df[~df['is_peak_hour']].groupby('hour').size().mean()
    
    print(f"Peak hour average (6-10 PM): {peak_avg:,.0f} requests/hour")
    print(f"Off-peak average: {offpeak_avg:,.0f} requests/hour")
    print(f"Peak to off-peak ratio: {peak_avg/offpeak_avg:.2f}x")
    
    # Visualization: Requests by hour
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#e74c3c' if 18 <= h <= 22 else '#3498db' for h in hourly_requests['hour']]
    bars = ax.bar(hourly_requests['hour'], hourly_requests['requests'], color=colors, edgecolor='white')
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Total Requests', fontsize=12)
    ax.set_title('Traffic Distribution by Hour of Day', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Peak Hours (6-10 PM)'),
                       Patch(facecolor='#3498db', label='Off-Peak Hours')]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add average line
    ax.axhline(hourly_requests['requests'].mean(), color='#2c3e50', linestyle='--', 
               label=f'Average: {hourly_requests["requests"].mean():,.0f}')
    
    plt.tight_layout()
    save_figure(fig, 'requests_by_hour.png', viz_dir)
    
    return hourly_requests


def analyze_response_times(df: pd.DataFrame, viz_dir: str) -> pd.DataFrame:
    """
    Analyze and visualize response times.
    Computes average, median, P95, and P99 response times.
    """
    print("\n--- Response Time Analysis ---")
    
    # Overall statistics
    stats = {
        'mean': df['response_time_ms'].mean(),
        'median': df['response_time_ms'].median(),
        'std': df['response_time_ms'].std(),
        'p50': df['response_time_ms'].quantile(0.50),
        'p90': df['response_time_ms'].quantile(0.90),
        'p95': df['response_time_ms'].quantile(0.95),
        'p99': df['response_time_ms'].quantile(0.99),
        'max': df['response_time_ms'].max()
    }
    
    print(f"Mean response time: {stats['mean']:.1f}ms")
    print(f"Median response time: {stats['median']:.1f}ms")
    print(f"P95 response time: {stats['p95']:.1f}ms")
    print(f"P99 response time: {stats['p99']:.1f}ms")
    print(f"Max response time: {stats['max']:.1f}ms")
    
    # Response time by hour
    hourly_response = df.groupby('hour').agg({
        'response_time_ms': ['mean', 'median', lambda x: x.quantile(0.95)]
    }).round(2)
    hourly_response.columns = ['mean_ms', 'median_ms', 'p95_ms']
    hourly_response = hourly_response.reset_index()
    
    # Visualization 1: Response time distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(df['response_time_ms'], bins=50, edgecolor='white', alpha=0.7, color='#3498db')
    ax1.axvline(stats['mean'], color='#e74c3c', linestyle='-', linewidth=2, label=f'Mean: {stats["mean"]:.0f}ms')
    ax1.axvline(stats['p95'], color='#f39c12', linestyle='--', linewidth=2, label=f'P95: {stats["p95"]:.0f}ms')
    ax1.set_xlabel('Response Time (ms)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, stats['p99'] * 1.5)  # Limit x-axis for readability
    
    # Response time by hour
    ax2 = axes[1]
    ax2.plot(hourly_response['hour'], hourly_response['mean_ms'], marker='o', label='Mean', linewidth=2)
    ax2.plot(hourly_response['hour'], hourly_response['p95_ms'], marker='s', label='P95', linewidth=2)
    ax2.fill_between(hourly_response['hour'], hourly_response['mean_ms'], 
                     hourly_response['p95_ms'], alpha=0.2)
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Response Time (ms)', fontsize=12)
    ax2.set_title('Response Time by Hour', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend()
    
    # Highlight peak hours
    ax2.axvspan(18, 22, alpha=0.1, color='red', label='Peak Hours')
    
    plt.tight_layout()
    save_figure(fig, 'response_times.png', viz_dir)
    
    return hourly_response


def analyze_error_rates(df: pd.DataFrame, viz_dir: str) -> pd.DataFrame:
    """
    Analyze and visualize error rates over time.
    """
    print("\n--- Error Rate Analysis ---")
    
    # Overall error rate
    overall_error_rate = df['error'].mean() * 100
    print(f"Overall error rate: {overall_error_rate:.2f}%")
    
    # Error rate by hour
    hourly_errors = df.groupby('hour').agg({
        'error': 'mean'
    }).reset_index()
    hourly_errors['error_rate'] = hourly_errors['error'] * 100
    
    # Error rate by date
    daily_errors = df.groupby('date').agg({
        'error': 'mean',
        'timestamp': 'count'
    }).reset_index()
    daily_errors.columns = ['date', 'error_rate', 'total_requests']
    daily_errors['error_rate'] = daily_errors['error_rate'] * 100
    
    # Error rate by status code distribution
    error_df = df[df['error']]
    status_distribution = error_df['status_code'].value_counts()
    
    print(f"\nError distribution by status code:")
    for code, count in status_distribution.items():
        pct = count / len(error_df) * 100
        print(f"  {code}: {count:,} ({pct:.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error rate by hour
    ax1 = axes[0, 0]
    colors = ['#e74c3c' if rate > overall_error_rate else '#27ae60' 
              for rate in hourly_errors['error_rate']]
    ax1.bar(hourly_errors['hour'], hourly_errors['error_rate'], color=colors, edgecolor='white')
    ax1.axhline(overall_error_rate, color='#2c3e50', linestyle='--', 
                label=f'Average: {overall_error_rate:.2f}%')
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Error Rate (%)', fontsize=12)
    ax1.set_title('Error Rate by Hour', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, 24, 2))
    ax1.legend()
    
    # 2. Error rate by date
    ax2 = axes[0, 1]
    ax2.plot(daily_errors['date'], daily_errors['error_rate'], marker='o', 
             linewidth=2, color='#e74c3c')
    ax2.fill_between(daily_errors['date'], daily_errors['error_rate'], alpha=0.3, color='#e74c3c')
    ax2.axhline(overall_error_rate, color='#2c3e50', linestyle='--')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('Daily Error Rate Trend', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Status code distribution (all requests)
    ax3 = axes[1, 0]
    status_counts = df['status_code'].value_counts()
    colors_status = ['#27ae60' if code == 200 else '#e74c3c' for code in status_counts.index]
    ax3.bar([str(c) for c in status_counts.index], status_counts.values, 
            color=colors_status, edgecolor='white')
    ax3.set_xlabel('Status Code', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Requests by Status Code', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Error status code distribution (pie chart)
    ax4 = axes[1, 1]
    colors_pie = plt.cm.Reds(np.linspace(0.3, 0.8, len(status_distribution)))
    wedges, texts, autotexts = ax4.pie(status_distribution.values, 
                                        labels=[str(c) for c in status_distribution.index],
                                        autopct='%1.1f%%', colors=colors_pie)
    ax4.set_title('Error Distribution by Status Code', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'error_analysis.png', viz_dir)
    
    return daily_errors


def analyze_endpoint_performance(df: pd.DataFrame, viz_dir: str) -> pd.DataFrame:
    """
    Analyze performance metrics by endpoint.
    Identifies bottlenecks and slow endpoints.
    """
    print("\n--- Endpoint Performance Analysis ---")
    
    # Calculate metrics by endpoint
    endpoint_stats = df.groupby(['service', 'endpoint']).agg({
        'response_time_ms': ['mean', 'median', lambda x: x.quantile(0.95), 'count'],
        'error': 'mean'
    }).round(2)
    
    endpoint_stats.columns = ['mean_ms', 'median_ms', 'p95_ms', 'request_count', 'error_rate']
    endpoint_stats['error_rate'] = endpoint_stats['error_rate'] * 100
    endpoint_stats = endpoint_stats.reset_index()
    endpoint_stats = endpoint_stats.sort_values('p95_ms', ascending=False)
    
    print("\nEndpoint Performance Summary (sorted by P95 latency):")
    print(endpoint_stats.to_string(index=False))
    
    # Calculate metrics by service
    service_stats = df.groupby('service').agg({
        'response_time_ms': ['mean', lambda x: x.quantile(0.95)],
        'error': 'mean',
        'timestamp': 'count'
    }).round(2)
    service_stats.columns = ['mean_ms', 'p95_ms', 'error_rate', 'total_requests']
    service_stats['error_rate'] = service_stats['error_rate'] * 100
    
    print("\nService-Level Summary:")
    print(service_stats.to_string())
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Response time by endpoint
    ax1 = axes[0, 0]
    endpoint_sorted = endpoint_stats.sort_values('mean_ms', ascending=True)
    y_pos = range(len(endpoint_sorted))
    ax1.barh(y_pos, endpoint_sorted['mean_ms'], color='#3498db', alpha=0.7, label='Mean')
    ax1.barh(y_pos, endpoint_sorted['p95_ms'], color='#e74c3c', alpha=0.5, label='P95')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{row['service']}{row['endpoint']}" 
                         for _, row in endpoint_sorted.iterrows()])
    ax1.set_xlabel('Response Time (ms)', fontsize=12)
    ax1.set_title('Response Time by Endpoint', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # 2. Error rate by endpoint
    ax2 = axes[0, 1]
    endpoint_by_error = endpoint_stats.sort_values('error_rate', ascending=True)
    y_pos = range(len(endpoint_by_error))
    colors = ['#e74c3c' if rate > 5 else '#f39c12' if rate > 3 else '#27ae60' 
              for rate in endpoint_by_error['error_rate']]
    ax2.barh(y_pos, endpoint_by_error['error_rate'], color=colors, edgecolor='white')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{row['service']}{row['endpoint']}" 
                         for _, row in endpoint_by_error.iterrows()])
    ax2.set_xlabel('Error Rate (%)', fontsize=12)
    ax2.set_title('Error Rate by Endpoint', fontsize=14, fontweight='bold')
    ax2.axvline(df['error'].mean() * 100, color='#2c3e50', linestyle='--', label='Average')
    ax2.legend()
    
    # 3. Request volume by service
    ax3 = axes[1, 0]
    service_colors = {'auth': '#3498db', 'events': '#27ae60', 'payments': '#e74c3c'}
    service_counts = df['service'].value_counts()
    ax3.pie(service_counts.values, labels=service_counts.index, autopct='%1.1f%%',
            colors=[service_colors.get(s, '#95a5a6') for s in service_counts.index],
            explode=[0.02] * len(service_counts))
    ax3.set_title('Request Distribution by Service', fontsize=14, fontweight='bold')
    
    # 4. Service comparison (grouped bar)
    ax4 = axes[1, 1]
    services = service_stats.index.tolist()
    x = np.arange(len(services))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, service_stats['mean_ms'], width, label='Mean RT (ms)', color='#3498db')
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, service_stats['error_rate'], width, label='Error Rate (%)', color='#e74c3c')
    
    ax4.set_xlabel('Service', fontsize=12)
    ax4.set_ylabel('Response Time (ms)', fontsize=12, color='#3498db')
    ax4_twin.set_ylabel('Error Rate (%)', fontsize=12, color='#e74c3c')
    ax4.set_title('Service Performance Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(services)
    
    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    save_figure(fig, 'endpoint_performance.png', viz_dir)
    
    return endpoint_stats


def analyze_peak_vs_offpeak(df: pd.DataFrame, viz_dir: str) -> pd.DataFrame:
    """
    Compare metrics during peak vs off-peak hours.
    """
    print("\n--- Peak vs Off-Peak Analysis ---")
    
    # Calculate metrics for peak and off-peak
    comparison = df.groupby('is_peak_hour').agg({
        'response_time_ms': ['mean', 'median', lambda x: x.quantile(0.95)],
        'error': 'mean',
        'timestamp': 'count'
    }).round(2)
    
    comparison.columns = ['mean_ms', 'median_ms', 'p95_ms', 'error_rate', 'request_count']
    comparison['error_rate'] = comparison['error_rate'] * 100
    comparison.index = ['Off-Peak', 'Peak']
    
    print("\nPeak (6-10 PM) vs Off-Peak Comparison:")
    print(comparison.to_string())
    
    # Calculate percentage differences
    peak = comparison.loc['Peak']
    offpeak = comparison.loc['Off-Peak']
    
    print(f"\nPerformance degradation during peak hours:")
    print(f"  Response time increase: {((peak['mean_ms']/offpeak['mean_ms'])-1)*100:.1f}%")
    print(f"  Error rate increase: {((peak['error_rate']/offpeak['error_rate'])-1)*100:.1f}%")
    print(f"  Traffic increase: {((peak['request_count']/offpeak['request_count'])-1)*100:.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    categories = ['Off-Peak', 'Peak']
    colors = ['#3498db', '#e74c3c']
    
    # Response time comparison
    ax1 = axes[0]
    x = np.arange(len(categories))
    width = 0.6
    ax1.bar(x, comparison['mean_ms'], width, color=colors, edgecolor='white')
    ax1.set_ylabel('Mean Response Time (ms)', fontsize=12)
    ax1.set_title('Response Time: Peak vs Off-Peak', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    for i, v in enumerate(comparison['mean_ms']):
        ax1.text(i, v + 2, f'{v:.0f}ms', ha='center', fontweight='bold')
    
    # Error rate comparison
    ax2 = axes[1]
    ax2.bar(x, comparison['error_rate'], width, color=colors, edgecolor='white')
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('Error Rate: Peak vs Off-Peak', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    for i, v in enumerate(comparison['error_rate']):
        ax2.text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold')
    
    # Request volume comparison
    ax3 = axes[2]
    ax3.bar(x, comparison['request_count'], width, color=colors, edgecolor='white')
    ax3.set_ylabel('Total Requests', fontsize=12)
    ax3.set_title('Traffic Volume: Peak vs Off-Peak', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    for i, v in enumerate(comparison['request_count']):
        ax3.text(i, v + 1000, f'{v:,.0f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'peak_vs_offpeak.png', viz_dir)
    
    return comparison


def create_summary_dashboard(df: pd.DataFrame, viz_dir: str) -> None:
    """
    Create a summary dashboard with key metrics.
    """
    print("\n--- Creating Summary Dashboard ---")
    
    # Calculate key metrics
    total_requests = len(df)
    total_days = df['date'].nunique()
    avg_daily_requests = total_requests / total_days
    overall_error_rate = df['error'].mean() * 100
    mean_rt = df['response_time_ms'].mean()
    p95_rt = df['response_time_ms'].quantile(0.95)
    
    # Create dashboard
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('System Analytics Dashboard', fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Daily request trend
    ax1 = axes[0, 0]
    daily = df.groupby('date').size()
    ax1.plot(daily.index, daily.values, marker='o', linewidth=2, color='#3498db')
    ax1.fill_between(daily.index, daily.values, alpha=0.3, color='#3498db')
    ax1.set_title(f'Daily Requests\n(Avg: {avg_daily_requests:,.0f}/day)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Requests')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Hourly pattern
    ax2 = axes[0, 1]
    hourly = df.groupby('hour').size()
    colors = ['#e74c3c' if 18 <= h <= 22 else '#3498db' for h in hourly.index]
    ax2.bar(hourly.index, hourly.values, color=colors, edgecolor='white')
    ax2.set_title('Traffic by Hour', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Requests')
    
    # 3. Service distribution
    ax3 = axes[0, 2]
    service_counts = df['service'].value_counts()
    ax3.pie(service_counts.values, labels=service_counts.index, autopct='%1.1f%%',
            colors=['#3498db', '#27ae60', '#e74c3c'])
    ax3.set_title('Requests by Service', fontsize=12, fontweight='bold')
    
    # 4. Response time trend
    ax4 = axes[1, 0]
    daily_rt = df.groupby('date')['response_time_ms'].agg(['mean', lambda x: x.quantile(0.95)])
    daily_rt.columns = ['mean', 'p95']
    ax4.plot(daily_rt.index, daily_rt['mean'], marker='o', label='Mean', color='#3498db')
    ax4.plot(daily_rt.index, daily_rt['p95'], marker='s', label='P95', color='#e74c3c')
    ax4.set_title(f'Response Time Trend\n(Avg: {mean_rt:.0f}ms, P95: {p95_rt:.0f}ms)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Response Time (ms)')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Error rate trend
    ax5 = axes[1, 1]
    daily_error = df.groupby('date')['error'].mean() * 100
    ax5.plot(daily_error.index, daily_error.values, marker='o', linewidth=2, color='#e74c3c')
    ax5.fill_between(daily_error.index, daily_error.values, alpha=0.3, color='#e74c3c')
    ax5.axhline(overall_error_rate, color='#2c3e50', linestyle='--')
    ax5.set_title(f'Daily Error Rate\n(Avg: {overall_error_rate:.2f}%)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Error Rate (%)')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Key metrics summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    metrics_text = f"""
    KEY METRICS SUMMARY
    ========================
    
    Total Requests: {total_requests:,}
    Analysis Period: {total_days} days
    Daily Average: {avg_daily_requests:,.0f}
    
    Mean Response: {mean_rt:.0f}ms
    P95 Response: {p95_rt:.0f}ms
    
    Error Rate: {overall_error_rate:.2f}%
    Total Errors: {df['error'].sum():,}
    
    Busiest Hour: {df.groupby('hour').size().idxmax()}:00
    Quietest Hour: {df.groupby('hour').size().idxmin()}:00
    """
    ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.tight_layout()
    save_figure(fig, 'dashboard.png', viz_dir)


def save_summary_tables(df: pd.DataFrame, endpoint_stats: pd.DataFrame, 
                        daily_errors: pd.DataFrame, output_dir: str) -> None:
    """Save summary tables as CSV files."""
    print("\n--- Saving Summary Tables ---")
    
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # Endpoint performance summary
    endpoint_stats.to_csv(os.path.join(tables_dir, 'endpoint_performance.csv'), index=False)
    print(f"  Saved: endpoint_performance.csv")
    
    # Daily metrics
    daily_metrics = df.groupby('date').agg({
        'response_time_ms': ['mean', 'median', lambda x: x.quantile(0.95)],
        'error': ['mean', 'sum'],
        'timestamp': 'count'
    }).round(2)
    daily_metrics.columns = ['mean_rt_ms', 'median_rt_ms', 'p95_rt_ms', 
                             'error_rate', 'error_count', 'total_requests']
    daily_metrics['error_rate'] = daily_metrics['error_rate'] * 100
    daily_metrics.to_csv(os.path.join(tables_dir, 'daily_metrics.csv'))
    print(f"  Saved: daily_metrics.csv")
    
    # Hourly metrics
    hourly_metrics = df.groupby('hour').agg({
        'response_time_ms': ['mean', lambda x: x.quantile(0.95)],
        'error': 'mean',
        'timestamp': 'count'
    }).round(2)
    hourly_metrics.columns = ['mean_rt_ms', 'p95_rt_ms', 'error_rate', 'total_requests']
    hourly_metrics['error_rate'] = hourly_metrics['error_rate'] * 100
    hourly_metrics.to_csv(os.path.join(tables_dir, 'hourly_metrics.csv'))
    print(f"  Saved: hourly_metrics.csv")


def main():
    """Main analysis function."""
    print("=" * 60)
    print("SYSTEM USAGE & RELIABILITY ANALYTICS")
    print("=" * 60)
    
    # Paths
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '..', 'data', 'processed', 'clean_logs.csv')
    viz_dir = os.path.join(script_dir, '..', 'visualizations')
    
    # Ensure visualization directory exists
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load data
    df = load_clean_data(data_path)
    
    # Run analyses
    hourly_requests = analyze_requests_per_hour(df, viz_dir)
    hourly_response = analyze_response_times(df, viz_dir)
    daily_errors = analyze_error_rates(df, viz_dir)
    endpoint_stats = analyze_endpoint_performance(df, viz_dir)
    peak_comparison = analyze_peak_vs_offpeak(df, viz_dir)
    
    # Create dashboard
    create_summary_dashboard(df, viz_dir)
    
    # Save summary tables
    save_summary_tables(df, endpoint_stats, daily_errors, viz_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nVisualizations saved to: {viz_dir}")
    print("\nGenerated files:")
    print("  - requests_by_hour.png")
    print("  - response_times.png")
    print("  - error_analysis.png")
    print("  - endpoint_performance.png")
    print("  - peak_vs_offpeak.png")
    print("  - dashboard.png")
    print("  - tables/endpoint_performance.csv")
    print("  - tables/daily_metrics.csv")
    print("  - tables/hourly_metrics.csv")


if __name__ == '__main__':
    main()
