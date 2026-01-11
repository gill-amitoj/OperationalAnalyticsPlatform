"""
create_dashboard.py
===================
Creates an interactive HTML dashboard using Plotly.
This generates a single HTML file that can be opened in any browser
or hosted on GitHub Pages.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """Load cleaned data."""
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '..', 'data', 'processed', 'clean_logs.csv')
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df):,} records")
    return df

# =============================================================================
# CREATE VISUALIZATIONS
# =============================================================================

def create_traffic_by_hour(df):
    """Create hourly traffic bar chart."""
    # Ensure all 24 hours are present
    all_hours = pd.DataFrame({'hour': list(range(24))})
    hourly = df.groupby('hour').size().reset_index(name='requests')
    hourly = all_hours.merge(hourly, on='hour', how='left').fillna(0)
    hourly['requests'] = hourly['requests'].astype(int)
    # Color peak hours differently
    hourly['period'] = hourly['hour'].apply(
        lambda h: 'Peak (6-10 PM)' if 18 <= h <= 22 else 'Off-Peak'
    )
    fig = px.bar(
        hourly, 
        x='hour', 
        y='requests',
        color='period',
        color_discrete_map={'Peak (6-10 PM)': '#e74c3c', 'Off-Peak': '#3498db'},
        title='<b>Traffic Distribution by Hour</b>',
        labels={'hour': 'Hour of Day', 'requests': 'Total Requests', 'period': 'Period'}
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        hovermode='x unified'
    )
    return fig


def create_response_time_trend(df):
    """Create daily response time trend."""
    daily = df.groupby('date').agg({
        'response_time_ms': ['mean', 'median', lambda x: x.quantile(0.95)]
    }).round(2)
    daily.columns = ['Mean', 'Median', 'P95']
    daily = daily.reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['Mean'],
        name='Mean', mode='lines+markers',
        line=dict(color='#3498db', width=2),
        hovertemplate='Mean: %{y:.0f}ms<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['P95'],
        name='P95', mode='lines+markers',
        line=dict(color='#e74c3c', width=2),
        hovertemplate='P95: %{y:.0f}ms<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['Median'],
        name='Median', mode='lines+markers',
        line=dict(color='#27ae60', width=2),
        hovertemplate='Median: %{y:.0f}ms<extra></extra>'
    ))
    
    fig.update_layout(
        title='<b>Response Time Trend (Daily)</b>',
        xaxis_title='Date',
        yaxis_title='Response Time (ms)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig


def create_error_rate_heatmap(df):
    """Create error rate heatmap by hour and day."""
    # Ensure all days and hours are present
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    all_hours = list(range(24))
    all_combos = pd.MultiIndex.from_product([day_order, all_hours], names=['day_of_week', 'hour'])
    heatmap_data = df.groupby(['day_of_week', 'hour']).agg({'error': 'mean'}).reindex(all_combos, fill_value=0).reset_index()
    heatmap_data['error_rate'] = heatmap_data['error'] * 100
    pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='error_rate').reindex(day_order)
    fig = px.imshow(
        pivot,
        labels=dict(x='Hour of Day', y='Day of Week', color='Error Rate (%)'),
        title='<b>Error Rate Heatmap by Hour and Day</b>',
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    return fig


def create_endpoint_performance(df):
    """Create endpoint performance comparison."""
    endpoint_stats = df.groupby(['service', 'endpoint']).agg({
        'response_time_ms': ['mean', lambda x: x.quantile(0.95)],
        'error': 'mean',
        'timestamp': 'count'
    }).round(2)
    
    endpoint_stats.columns = ['mean_ms', 'p95_ms', 'error_rate', 'requests']
    endpoint_stats['error_rate'] = endpoint_stats['error_rate'] * 100
    endpoint_stats = endpoint_stats.reset_index()
    endpoint_stats['full_endpoint'] = endpoint_stats['service'] + endpoint_stats['endpoint']
    endpoint_stats = endpoint_stats.sort_values('mean_ms', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=endpoint_stats['full_endpoint'],
        x=endpoint_stats['mean_ms'],
        name='Mean',
        orientation='h',
        marker_color='#3498db',
        hovertemplate='Mean: %{x:.0f}ms<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=endpoint_stats['full_endpoint'],
        x=endpoint_stats['p95_ms'],
        name='P95',
        orientation='h',
        marker_color='#e74c3c',
        opacity=0.7,
        hovertemplate='P95: %{x:.0f}ms<extra></extra>'
    ))
    
    fig.update_layout(
        title='<b>Response Time by Endpoint</b>',
        xaxis_title='Response Time (ms)',
        yaxis_title='Endpoint',
        barmode='overlay',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=500
    )
    
    return fig


def create_service_distribution(df):
    """Create service request distribution pie chart."""
    # Use actual counts, but ensure all services are present for completeness
    print("[DEBUG] Service value counts before pie chart:")
    print(df['service'].value_counts())
    all_services = ['auth', 'events', 'payments']
    service_counts = df['service'].value_counts().reindex(all_services, fill_value=0)
    service_counts = service_counts.reset_index()
    service_counts.columns = ['service', 'requests']
    fig = px.pie(
        service_counts,
        values='requests',
        names='service',
        title='<b>Request Distribution by Service</b>',
        color='service',
        color_discrete_map={
            'auth': '#3498db',
            'events': '#27ae60', 
            'payments': '#e74c3c'
        },
        hole=0.4
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}<br>Requests: %{value:,}<br>Percentage: %{percent}<extra></extra>'
    )
    # Add warning if all values are equal (flat pie)
    if len(service_counts['requests'].unique()) == 1:
        fig.update_layout(
            annotations=[dict(text='Warning: Flat distribution! Check data.', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
    return fig


def create_error_by_service(df):
    """Create error rate by service bar chart."""
    # Ensure all services are present
    all_services = ['auth', 'events', 'payments']
    service_errors = df.groupby('service').agg({'error': 'mean', 'timestamp': 'count'}).reindex(all_services, fill_value=0).reset_index()
    service_errors.columns = ['service', 'error_rate', 'total_requests']
    service_errors['error_rate'] = service_errors['error_rate'] * 100
    fig = px.bar(
        service_errors,
        x='service',
        y='error_rate',
        color='service',
        color_discrete_map={
            'auth': '#3498db',
            'events': '#27ae60',
            'payments': '#e74c3c'
        },
        title='<b>Error Rate by Service</b>',
        labels={'error_rate': 'Error Rate (%)', 'service': 'Service'},
        text=service_errors['error_rate'].apply(lambda x: f'{x:.2f}%')
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)
    return fig


def create_status_code_distribution(df):
    """Create status code distribution."""
    # Ensure all common status codes are present
    all_codes = ['200', '400', '401', '403', '404', '500', '502', '503']
    status_counts = df['status_code'].astype(str).value_counts().reindex(all_codes, fill_value=0).reset_index()
    status_counts.columns = ['status_code', 'count']
    # Color by success/error
    status_counts['type'] = status_counts['status_code'].apply(
        lambda x: 'Success' if x.startswith('2') else 'Error'
    )
    fig = px.bar(
        status_counts,
        x='status_code',
        y='count',
        color='type',
        color_discrete_map={'Success': '#27ae60', 'Error': '#e74c3c'},
        title='<b>Response Status Code Distribution</b>',
        labels={'status_code': 'Status Code', 'count': 'Count', 'type': 'Type'}
    )
    return fig


def create_peak_comparison(df):
    """Create peak vs off-peak comparison."""
    comparison = df.groupby('is_peak_hour').agg({
        'response_time_ms': 'mean',
        'error': 'mean'
    }).reset_index()
    comparison['error'] = comparison['error'] * 100
    comparison['period'] = comparison['is_peak_hour'].map({False: 'Off-Peak', True: 'Peak (6-10 PM)'})
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Response Time', 'Error Rate'))
    
    fig.add_trace(
        go.Bar(
            x=comparison['period'],
            y=comparison['response_time_ms'],
            marker_color=['#3498db', '#e74c3c'],
            text=comparison['response_time_ms'].apply(lambda x: f'{x:.0f}ms'),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=comparison['period'],
            y=comparison['error'],
            marker_color=['#3498db', '#e74c3c'],
            text=comparison['error'].apply(lambda x: f'{x:.2f}%'),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='<b>Peak vs Off-Peak Performance</b>',
        height=400
    )
    
    fig.update_yaxes(title_text='Response Time (ms)', row=1, col=1)
    fig.update_yaxes(title_text='Error Rate (%)', row=1, col=2)
    
    return fig


def calculate_kpis(df):
    """Calculate key performance indicators."""
    return {
        'total_requests': len(df),
        'total_days': df['date'].nunique(),
        'avg_daily': len(df) / df['date'].nunique(),
        'mean_rt': df['response_time_ms'].mean(),
        'p95_rt': df['response_time_ms'].quantile(0.95),
        'error_rate': df['error'].mean() * 100,
        'total_errors': df['error'].sum(),
        'peak_ratio': df[df['is_peak_hour']].groupby('hour').size().mean() / df[~df['is_peak_hour']].groupby('hour').size().mean()
    }


def create_full_dashboard(df):
    """Create the full interactive dashboard."""
    
    kpis = calculate_kpis(df)
    
    # Create all figures
    fig_traffic = create_traffic_by_hour(df)
    fig_response = create_response_time_trend(df)
    fig_heatmap = create_error_rate_heatmap(df)
    fig_endpoint = create_endpoint_performance(df)
    # fig_service = create_service_distribution(df)
    fig_error_service = create_error_by_service(df)
    fig_status = create_status_code_distribution(df)
    fig_peak = create_peak_comparison(df)
    
    # Build HTML with static PNGs embedded
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>System Analytics Dashboard - Updated {timestamp}</title>
    <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
    <style>
        body {{
            font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
            background: #f7f9fa;
            color: #222;
            margin: 0;
        }}
        .header {{
            background: #fff;
            padding: 32px 0 16px 0;
            text-align: center;
            box-shadow: 0 2px 8px #eaeaea;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 2.5rem;
            font-weight: 700;
            color: #2d3e50;
        }}
        .header p {{
            margin: 0;
            color: #888;
            font-size: 1.1rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 32px auto 0 auto;
            padding: 0 24px 32px 24px;
        }}
        .kpi-grid {{
            display: flex;
            gap: 24px;
            margin-bottom: 32px;
            flex-wrap: wrap;
        }}
        .kpi-card {{
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 1px 6px #eaeaea;
            padding: 24px 32px;
            flex: 1 1 180px;
            min-width: 180px;
            text-align: center;
        }}
        .kpi-label {{
            color: #888;
            font-size: 1rem;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 2.1rem;
            font-weight: 600;
            color: #2d3e50;
        }}
        .section-title {{
            font-size: 1.5rem;
            font-weight: 600;
            margin: 40px 0 18px 0;
            color: #2d3e50;
        }}
        .chart-grid {{
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            margin-bottom: 32px;
        }}
        .chart-card {{
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 1px 6px #eaeaea;
            padding: 18px 18px 8px 18px;
            flex: 1 1 340px;
            min-width: 320px;
            margin-bottom: 12px;
        }}
        .chart-card.full-width {{
            flex-basis: 100%;
            min-width: 0;
        }}
        .footer {{
            background: #fff;
            text-align: center;
            padding: 24px 0 12px 0;
            color: #aaa;
            font-size: 1rem;
            margin-top: 32px;
            border-top: 1px solid #eee;
        }}
        @media (max-width: 900px) {{
            .chart-grid, .kpi-grid {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>System Usage & Reliability Analytics</h1>
        <p>Interactive Dashboard &mdash; 10 Days of System Log Analysis (Synthetic Data)</p>
        <p style='color:#888;font-size:0.95rem;'>Last generated: {timestamp}</p>
    </div>
    <div class="container">
        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Requests</div>
                <div class="kpi-value">{kpis['total_requests']:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Avg. Daily Requests</div>
                <div class="kpi-value">{kpis['avg_daily']:.0f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Mean Response Time</div>
                <div class="kpi-value">{kpis['mean_rt']:.0f} ms</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">P95 Response Time</div>
                <div class="kpi-value">{kpis['p95_rt']:.0f} ms</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Error Rate</div>
                <div class="kpi-value">{kpis['error_rate']:.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Errors</div>
                <div class="kpi-value">{kpis['total_errors']:,}</div>
            </div>
        </div>
        <!-- Key Insights -->
        <div style="margin-bottom: 32px;">
            <ul style="font-size:1.08rem; color:#2d3e50; line-height:1.7;">
                <li><b>Peak hours</b> (6-10 PM) see a {kpis['peak_ratio']:.2f}x increase in traffic compared to off-peak.</li>
                <li><b>Response times</b> are generally stable, with P95 below 1s for most days.</li>
                <li><b>Error rates</b> remain low, with spikes during peak load and for certain endpoints.</li>
                <li>All data and visualizations are <b>synthetically generated</b> for demonstration.</li>
            </ul>
        </div>
        <h2 class="section-title">Traffic Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <div id="traffic-chart"></div>
                <img src="../visualizations/requests_by_hour.png" alt="Requests by Hour" style="width:100%;margin-top:10px;border-radius:8px;box-shadow:0 1px 6px #ccc;">
            </div>
            <div class="chart-card">
                <img src="../visualizations/dashboard.png" alt="Dashboard Summary" style="width:100%;margin-top:10px;border-radius:8px;box-shadow:0 1px 6px #ccc;">
            </div>
        </div>
        <h2 class="section-title">Performance Metrics</h2>
        <div class="chart-grid">
            <div class="chart-card full-width">
                <div id="response-chart"></div>
                <img src="../visualizations/response_times.png" alt="Response Times" style="width:100%;margin-top:10px;border-radius:8px;box-shadow:0 1px 6px #ccc;">
            </div>
            <div class="chart-card">
                <div id="endpoint-chart"></div>
                <img src="../visualizations/endpoint_performance.png" alt="Endpoint Performance" style="width:100%;margin-top:10px;border-radius:8px;box-shadow:0 1px 6px #ccc;">
            </div>
            <div class="chart-card">
                <div id="peak-chart"></div>
                <img src="../visualizations/peak_vs_offpeak.png" alt="Peak vs Off-Peak" style="width:100%;margin-top:10px;border-radius:8px;box-shadow:0 1px 6px #ccc;">
            </div>
        </div>
        <h2 class="section-title">Error Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card full-width">
                <div id="heatmap-chart"></div>
                <img src="../visualizations/error_analysis.png" alt="Error Analysis" style="width:100%;margin-top:10px;border-radius:8px;box-shadow:0 1px 6px #ccc;">
            </div>
            <div class="chart-card">
                <div id="status-chart"></div>
            </div>
            <div class="chart-card">
                <div id="error-service-chart"></div>
            </div>
        </div>
    </div>
    <div class="footer">
        <p>Built with Python, Pandas &amp; Plotly &mdash; Data is synthetically generated</p>
        <p style="margin-top: 10px;">&copy; 2026 System Analytics Project</p>
    </div>
    <script>
        // Render Plotly charts
        var traffic_fig = {fig_traffic.to_json()};
        var response_fig = {fig_response.to_json()};
        var heatmap_fig = {fig_heatmap.to_json()};
        var endpoint_fig = {fig_endpoint.to_json()};
        
        var error_service_fig = {fig_error_service.to_json()};
        var status_fig = {fig_status.to_json()};
        var peak_fig = {fig_peak.to_json()};
    Plotly.newPlot('traffic-chart', traffic_fig.data, traffic_fig.layout, {{responsive: true}});
    Plotly.newPlot('response-chart', response_fig.data, response_fig.layout, {{responsive: true}});
    Plotly.newPlot('heatmap-chart', heatmap_fig.data, heatmap_fig.layout, {{responsive: true}});
    Plotly.newPlot('endpoint-chart', endpoint_fig.data, endpoint_fig.layout, {{responsive: true}});
    // Plotly.newPlot('service-chart', service_fig.data, service_fig.layout, {{responsive: true}});
    Plotly.newPlot('error-service-chart', error_service_fig.data, error_service_fig.layout, {{responsive: true}});
    Plotly.newPlot('status-chart', status_fig.data, status_fig.layout, {{responsive: true}});
    Plotly.newPlot('peak-chart', peak_fig.data, peak_fig.layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    return html_content


def main():
    """Main function to create dashboard."""
    print("Creating Interactive Dashboard...")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Create dashboard HTML
    html_content = create_full_dashboard(df)
    
    # Save to file
    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, '..', 'visualizations', 'dashboard.html')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Dashboard saved to: {output_path}")
    print("\nOpen the HTML file in your browser to view the interactive dashboard!")
    print("You can also host it on GitHub Pages for easy sharing.")


if __name__ == '__main__':
    main()
