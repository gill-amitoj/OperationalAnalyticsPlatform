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
    hourly = df.groupby('hour').size().reset_index(name='requests')
    
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
    # Calculate error rate by hour and day of week
    heatmap_data = df.groupby(['day_of_week', 'hour']).agg({
        'error': 'mean'
    }).reset_index()
    heatmap_data['error_rate'] = heatmap_data['error'] * 100
    
    # Pivot for heatmap
    pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='error_rate')
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(day_order)
    
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
    service_counts = df['service'].value_counts().reset_index()
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
    
    return fig


def create_error_by_service(df):
    """Create error rate by service bar chart."""
    service_errors = df.groupby('service').agg({
        'error': 'mean',
        'timestamp': 'count'
    }).reset_index()
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
    status_counts = df['status_code'].value_counts().reset_index()
    status_counts.columns = ['status_code', 'count']
    status_counts['status_code'] = status_counts['status_code'].astype(str)
    
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
    fig_service = create_service_distribution(df)
    fig_error_service = create_error_by_service(df)
    fig_status = create_status_code_distribution(df)
    fig_peak = create_peak_comparison(df)
    
    # Build HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Analytics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }}
        
        .header {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 30px 40px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 30px;
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .kpi-card {{
            background: linear-gradient(145deg, #1e3a5f 0%, #1a2d47 100%);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
        }}
        
        .kpi-value {{
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .kpi-label {{
            font-size: 0.9rem;
            color: #a0aec0;
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 25px;
        }}
        
        .chart-card {{
            background: linear-gradient(145deg, #1e3a5f 0%, #1a2d47 100%);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .chart-card.full-width {{
            grid-column: span 2;
        }}
        
        .section-title {{
            font-size: 1.5rem;
            margin: 40px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
            display: inline-block;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: #a0aec0;
            font-size: 0.9rem;
        }}
        
        .footer a {{
            color: #667eea;
            text-decoration: none;
        }}
        
        .insights {{
            background: linear-gradient(145deg, #1e3a5f 0%, #1a2d47 100%);
            border-radius: 16px;
            padding: 25px;
            margin: 30px 0;
            border-left: 4px solid #667eea;
        }}
        
        .insights h3 {{
            margin-bottom: 15px;
            color: #667eea;
        }}
        
        .insights ul {{
            list-style: none;
            padding: 0;
        }}
        
        .insights li {{
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }}
        
        .insights li::before {{
            content: "→";
            position: absolute;
            left: 0;
            color: #667eea;
        }}
        
        @media (max-width: 1200px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
            .chart-card.full-width {{
                grid-column: span 1;
            }}
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8rem;
            }}
            .kpi-value {{
                font-size: 1.8rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 System Usage & Reliability Analytics</h1>
        <p>Interactive Dashboard | 10 Days of System Log Analysis | Synthetic Data</p>
    </div>
    
    <div class="container">
        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">{kpis['total_requests']:,}</div>
                <div class="kpi-label">Total Requests</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{kpis['avg_daily']:,.0f}</div>
                <div class="kpi-label">Daily Average</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{kpis['mean_rt']:.0f}ms</div>
                <div class="kpi-label">Mean Response Time</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{kpis['p95_rt']:.0f}ms</div>
                <div class="kpi-label">P95 Response Time</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{kpis['error_rate']:.2f}%</div>
                <div class="kpi-label">Error Rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{kpis['peak_ratio']:.1f}x</div>
                <div class="kpi-label">Peak Traffic Ratio</div>
            </div>
        </div>
        
        <!-- Key Insights -->
        <div class="insights">
            <h3>📊 Key Insights</h3>
            <ul>
                <li><strong>Peak Hours Impact:</strong> Traffic during 6-10 PM is {kpis['peak_ratio']:.1f}x higher with ~50% slower response times</li>
                <li><strong>Performance Bottleneck:</strong> Payments service shows highest latency (checkout endpoint: ~400ms avg)</li>
                <li><strong>Reliability:</strong> Overall error rate of {kpis['error_rate']:.2f}% with {kpis['total_errors']:,} total errors</li>
                <li><strong>Recommendation:</strong> Consider scaling resources during peak hours and optimizing payment endpoints</li>
            </ul>
        </div>
        
        <h2 class="section-title">📈 Traffic Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <div id="traffic-chart"></div>
            </div>
            <div class="chart-card">
                <div id="service-chart"></div>
            </div>
        </div>
        
        <h2 class="section-title">⚡ Performance Metrics</h2>
        <div class="chart-grid">
            <div class="chart-card full-width">
                <div id="response-chart"></div>
            </div>
            <div class="chart-card">
                <div id="endpoint-chart"></div>
            </div>
            <div class="chart-card">
                <div id="peak-chart"></div>
            </div>
        </div>
        
        <h2 class="section-title">🚨 Error Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card full-width">
                <div id="heatmap-chart"></div>
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
        <p>Built with Python, Pandas & Plotly | Data is synthetically generated</p>
        <p style="margin-top: 10px;">© 2026 System Analytics Project</p>
    </div>
    
    <script>
        // Chart configurations
        const darkLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#a0aec0' }},
            margin: {{ t: 50, r: 20, b: 50, l: 60 }},
            xaxis: {{ gridcolor: 'rgba(255,255,255,0.1)' }},
            yaxis: {{ gridcolor: 'rgba(255,255,255,0.1)' }}
        }};
        
        const config = {{ responsive: true, displayModeBar: true }};
        
        // Traffic Chart
        const trafficData = {fig_traffic.to_json()};
        trafficData.layout = {{...trafficData.layout, ...darkLayout}};
        Plotly.newPlot('traffic-chart', trafficData.data, trafficData.layout, config);
        
        // Service Distribution
        const serviceData = {fig_service.to_json()};
        serviceData.layout = {{...serviceData.layout, ...darkLayout}};
        Plotly.newPlot('service-chart', serviceData.data, serviceData.layout, config);
        
        // Response Time Trend
        const responseData = {fig_response.to_json()};
        responseData.layout = {{...responseData.layout, ...darkLayout}};
        Plotly.newPlot('response-chart', responseData.data, responseData.layout, config);
        
        // Endpoint Performance
        const endpointData = {fig_endpoint.to_json()};
        endpointData.layout = {{...endpointData.layout, ...darkLayout}};
        Plotly.newPlot('endpoint-chart', endpointData.data, endpointData.layout, config);
        
        // Peak Comparison
        const peakData = {fig_peak.to_json()};
        peakData.layout = {{...peakData.layout, ...darkLayout}};
        Plotly.newPlot('peak-chart', peakData.data, peakData.layout, config);
        
        // Error Heatmap
        const heatmapData = {fig_heatmap.to_json()};
        heatmapData.layout = {{...heatmapData.layout, ...darkLayout}};
        Plotly.newPlot('heatmap-chart', heatmapData.data, heatmapData.layout, config);
        
        // Status Code Distribution
        const statusData = {fig_status.to_json()};
        statusData.layout = {{...statusData.layout, ...darkLayout}};
        Plotly.newPlot('status-chart', statusData.data, statusData.layout, config);
        
        // Error by Service
        const errorServiceData = {fig_error_service.to_json()};
        errorServiceData.layout = {{...errorServiceData.layout, ...darkLayout}};
        Plotly.newPlot('error-service-chart', errorServiceData.data, errorServiceData.layout, config);
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
