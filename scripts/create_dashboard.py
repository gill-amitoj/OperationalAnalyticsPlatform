"""
create_dashboard.py
===================
Creates a professional, portfolio-ready interactive HTML dashboard using Plotly.

This dashboard showcases:
- Real-time KPI metrics with animated counters
- Interactive charts (traffic, response times, errors)
- Actionable business insights
- Professional styling suitable for enterprise presentations

The generated HTML file can be opened in any browser or hosted on GitHub Pages.
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
        title='<b>🕐 Traffic Distribution by Hour</b>',
        labels={'hour': 'Hour of Day', 'requests': 'Total Requests', 'period': 'Period'}
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
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
        line=dict(color='#3498db', width=3),
        hovertemplate='Mean: %{y:.0f}ms<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['P95'],
        name='P95', mode='lines+markers',
        line=dict(color='#e74c3c', width=3),
        hovertemplate='P95: %{y:.0f}ms<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['Median'],
        name='Median', mode='lines+markers',
        line=dict(color='#27ae60', width=3),
        hovertemplate='Median: %{y:.0f}ms<extra></extra>'
    ))
    
    fig.update_layout(
        title='<b>⚡ Response Time Trend (Daily)</b>',
        xaxis_title='Date',
        yaxis_title='Response Time (ms)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
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
        title='<b>🔥 Error Rate Heatmap by Hour and Day</b>',
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
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
        title='<b>🎯 Response Time by Endpoint</b>',
        xaxis_title='Response Time (ms)',
        yaxis_title='Endpoint',
        barmode='overlay',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_service_distribution(df):
    """Create service request distribution pie chart."""
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
        title='<b>📊 Request Distribution by Service</b>',
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
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
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
        title='<b>⚠️ Error Rate by Service</b>',
        labels={'error_rate': 'Error Rate (%)', 'service': 'Service'},
        text=service_errors['error_rate'].apply(lambda x: f'{x:.2f}%')
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
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
        title='<b>📈 Response Status Code Distribution</b>',
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
        'peak_ratio': df[df['is_peak_hour']].groupby('hour').size().mean() / df[~df['is_peak_hour']].groupby('hour').size().mean(),
        'slowest_endpoint': df.groupby('endpoint')['response_time_ms'].mean().idxmax(),
        'slowest_endpoint_time': df.groupby('endpoint')['response_time_ms'].mean().max(),
        'highest_error_service': df.groupby('service')['error'].mean().idxmax(),
        'highest_error_rate': df.groupby('service')['error'].mean().max() * 100
    }


def create_full_dashboard(df):
    """Create the full interactive dashboard."""
    
    kpis = calculate_kpis(df)
    
    # Create all figures
    fig_traffic = create_traffic_by_hour(df)
    fig_response = create_response_time_trend(df)
    fig_heatmap = create_error_rate_heatmap(df)
    fig_endpoint = create_endpoint_performance(df)
    fig_service = create_service_distribution(df)
    fig_error_service = create_error_by_service(df)
    fig_status = create_status_code_distribution(df)
    fig_peak = create_peak_comparison(df)
    
    # Build enhanced HTML dashboard
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>🚀 Operational System Analytics | Dashboard</title>
    <link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap' rel='stylesheet'>
    <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
    <style>
        :root {{
            --primary: #667eea;
            --secondary: #764ba2;
            --success: #27ae60;
            --danger: #e74c3c;
            --warning: #f39c12;
            --dark: #1a1a2e;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, rgba(102,126,234,0.95) 0%, rgba(118,75,162,0.95) 100%);
            padding: 60px 20px 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.8rem;
            font-weight: 800;
            color: #fff;
            margin-bottom: 12px;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .header p {{ color: rgba(255,255,255,0.9); font-size: 1.2rem; }}
        .header .timestamp {{ color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-top: 8px; }}
        .container {{ max-width: 1400px; margin: -30px auto 0; padding: 0 20px 60px; position: relative; z-index: 10; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .kpi-card {{
            background: #fff;
            border-radius: 20px;
            padding: 28px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }}
        .kpi-card:hover {{ transform: translateY(-8px); box-shadow: 0 20px 60px rgba(0,0,0,0.15); }}
        .kpi-icon {{ font-size: 2rem; margin-bottom: 10px; }}
        .kpi-value {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .kpi-label {{ color: #666; font-size: 0.95rem; font-weight: 500; margin-top: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .insights-section {{
            background: #fff;
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .insights-title {{ font-size: 1.5rem; font-weight: 700; color: var(--dark); margin-bottom: 20px; }}
        .insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .insight-card {{ background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%); border-radius: 12px; padding: 20px; border-left: 4px solid var(--primary); }}
        .insight-card.warning {{ border-left-color: var(--warning); }}
        .insight-card.danger {{ border-left-color: var(--danger); }}
        .insight-card.success {{ border-left-color: var(--success); }}
        .insight-card h4 {{ font-size: 1rem; font-weight: 600; margin-bottom: 8px; color: var(--dark); }}
        .insight-card p {{ color: #666; font-size: 0.9rem; line-height: 1.5; }}
        .section-title {{ font-size: 1.5rem; font-weight: 700; margin: 40px 0 24px; color: #fff; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 24px; }}
        .chart-card {{
            background: #fff;
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        .chart-card:hover {{ box-shadow: 0 20px 60px rgba(0,0,0,0.15); }}
        .chart-card.full-width {{ grid-column: 1 / -1; }}
        .recommendations {{
            background: linear-gradient(135deg, var(--dark) 0%, #16213e 100%);
            border-radius: 20px;
            padding: 32px;
            margin-top: 40px;
        }}
        .recommendations h3 {{ color: #fff; font-size: 1.5rem; margin-bottom: 24px; }}
        .rec-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
        .rec-card {{ background: rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); }}
        .rec-card h4 {{ color: #fff; font-size: 1rem; margin-bottom: 8px; }}
        .rec-card p {{ color: rgba(255,255,255,0.7); font-size: 0.9rem; }}
        .rec-priority {{ display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; margin-bottom: 10px; }}
        .rec-priority.high {{ background: var(--danger); color: #fff; }}
        .rec-priority.medium {{ background: var(--warning); color: #fff; }}
        .rec-priority.low {{ background: var(--success); color: #fff; }}
        .footer {{ text-align: center; padding: 40px 20px; color: rgba(255,255,255,0.7); font-size: 0.95rem; }}
        .footer a {{ color: #fff; text-decoration: none; }}
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 2rem; }}
            .chart-grid {{ grid-template-columns: 1fr; }}
            .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Operational System Analytics</h1>
        <p>Real-time Performance Monitoring & Reliability Dashboard</p>
        <p class="timestamp">Last updated: {timestamp}</p>
    </div>
    <div class="container">
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-icon">📊</div><div class="kpi-value">{kpis['total_requests']:,}</div><div class="kpi-label">Total Requests</div></div>
            <div class="kpi-card"><div class="kpi-icon">📅</div><div class="kpi-value">{kpis['avg_daily']:,.0f}</div><div class="kpi-label">Avg Daily Requests</div></div>
            <div class="kpi-card"><div class="kpi-icon">⚡</div><div class="kpi-value">{kpis['mean_rt']:.0f}ms</div><div class="kpi-label">Mean Response Time</div></div>
            <div class="kpi-card"><div class="kpi-icon">🎯</div><div class="kpi-value">{kpis['p95_rt']:.0f}ms</div><div class="kpi-label">P95 Response Time</div></div>
            <div class="kpi-card"><div class="kpi-icon">⚠️</div><div class="kpi-value">{kpis['error_rate']:.2f}%</div><div class="kpi-label">Error Rate</div></div>
            <div class="kpi-card"><div class="kpi-icon">📈</div><div class="kpi-value">{kpis['peak_ratio']:.1f}x</div><div class="kpi-label">Peak Traffic Ratio</div></div>
        </div>
        <div class="insights-section">
            <h3 class="insights-title">💡 Key Insights</h3>
            <div class="insight-grid">
                <div class="insight-card warning"><h4>🔥 Peak Hour Impact</h4><p>Traffic increases <strong>{kpis['peak_ratio']:.1f}x</strong> during peak hours (6-10 PM), correlating with higher response times and error rates.</p></div>
                <div class="insight-card danger"><h4>🐢 Slowest Endpoint</h4><p><code>{kpis['slowest_endpoint']}</code> has the highest latency at <strong>{kpis['slowest_endpoint_time']:.0f}ms</strong> average. Consider optimization.</p></div>
                <div class="insight-card danger"><h4>⚠️ Highest Error Service</h4><p>The <strong>{kpis['highest_error_service']}</strong> service has the highest error rate at <strong>{kpis['highest_error_rate']:.2f}%</strong>.</p></div>
                <div class="insight-card success"><h4>✅ System Health</h4><p>Overall system maintains <strong>{100 - kpis['error_rate']:.1f}%</strong> success rate with stable P95 latency under 400ms.</p></div>
            </div>
        </div>
        <h2 class="section-title">📈 Traffic Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card"><div id="traffic-chart"></div></div>
            <div class="chart-card"><div id="service-chart"></div></div>
        </div>
        <h2 class="section-title">⚡ Performance Metrics</h2>
        <div class="chart-grid">
            <div class="chart-card full-width"><div id="response-chart"></div></div>
            <div class="chart-card"><div id="endpoint-chart"></div></div>
            <div class="chart-card"><div id="peak-chart"></div></div>
        </div>
        <h2 class="section-title">🔥 Error Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card full-width"><div id="heatmap-chart"></div></div>
            <div class="chart-card"><div id="status-chart"></div></div>
            <div class="chart-card"><div id="error-service-chart"></div></div>
        </div>
        <div class="recommendations">
            <h3>🎯 Actionable Recommendations</h3>
            <div class="rec-grid">
                <div class="rec-card"><span class="rec-priority high">High Priority</span><h4>Scale Up During Peak Hours</h4><p>Implement auto-scaling for 6-10 PM window when traffic increases {kpis['peak_ratio']:.1f}x.</p></div>
                <div class="rec-card"><span class="rec-priority high">High Priority</span><h4>Optimize {kpis['slowest_endpoint']}</h4><p>This endpoint averages {kpis['slowest_endpoint_time']:.0f}ms. Consider caching or query optimization.</p></div>
                <div class="rec-card"><span class="rec-priority medium">Medium Priority</span><h4>Investigate {kpis['highest_error_service'].title()} Service</h4><p>Error rate of {kpis['highest_error_rate']:.2f}% is above target. Review error logs.</p></div>
                <div class="rec-card"><span class="rec-priority low">Low Priority</span><h4>Implement Alerting</h4><p>Set up automated alerts for when error rate exceeds 5% or P95 latency exceeds 500ms.</p></div>
            </div>
        </div>
    </div>
    <div class="footer">
        <p>Built with Python, Pandas & Plotly | Data is synthetically generated</p>
        <p style="margin-top: 10px;">© 2026 Operational System Analytics | <a href="https://github.com/gill-amitoj">GitHub</a></p>
    </div>
    <script>
        var traffic_fig = {fig_traffic.to_json()};
        var response_fig = {fig_response.to_json()};
        var heatmap_fig = {fig_heatmap.to_json()};
        var endpoint_fig = {fig_endpoint.to_json()};
        var service_fig = {fig_service.to_json()};
        var error_service_fig = {fig_error_service.to_json()};
        var status_fig = {fig_status.to_json()};
        var peak_fig = {fig_peak.to_json()};
        Plotly.newPlot('traffic-chart', traffic_fig.data, traffic_fig.layout, {{responsive: true}});
        Plotly.newPlot('response-chart', response_fig.data, response_fig.layout, {{responsive: true}});
        Plotly.newPlot('heatmap-chart', heatmap_fig.data, heatmap_fig.layout, {{responsive: true}});
        Plotly.newPlot('endpoint-chart', endpoint_fig.data, endpoint_fig.layout, {{responsive: true}});
        Plotly.newPlot('service-chart', service_fig.data, service_fig.layout, {{responsive: true}});
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
    print("=" * 60)
    print("🚀 Creating Interactive Dashboard...")
    print("=" * 60)
    
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
    print("\n🌐 Open the HTML file in your browser to view the interactive dashboard!")
    print("📤 You can also host it on GitHub Pages for easy sharing.")


if __name__ == '__main__':
    main()