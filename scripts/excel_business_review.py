"""
excel_business_review.py
=========================
Exports SQL-generated metrics to Excel for enterprise business review.

This script creates a multi-sheet Excel workbook with:
1. Metrics Sheet - Endpoint performance data from analysis
2. ServiceLookup Sheet - Reference table for VLOOKUP (service → owner, priority)
3. BusinessReview Sheet - Enriched data with formulas (VLOOKUP, IF, conditional logic)

This follows enterprise banking practices where analysts use Excel for:
- Executive reporting and stakeholder reviews
- Ad-hoc analysis with pivot tables
- SLA compliance tracking
- Service owner accountability

Usage:
    python scripts/excel_business_review.py
"""

import pandas as pd
import os


def export_sql_metrics_to_excel(metrics_path: str, excel_path: str) -> None:
    """
    Export SQL-generated metrics to Excel with business review formulas.
    
    Creates a professional Excel workbook with:
    - Raw metrics data
    - Lookup tables for service ownership
    - Business review sheet with VLOOKUP and IF formulas
    - Conditional formatting for SLA status
    
    Args:
        metrics_path: Path to the endpoint performance CSV
        excel_path: Output path for the Excel workbook
    """
    # Load metrics from SQL analysis output
    df_metrics = pd.read_csv(metrics_path)
    
    # Service lookup table (simulates enterprise service catalog)
    df_lookup = pd.DataFrame({
        'service': ['auth', 'events', 'payments'],
        'owner': ['Alice Chen', 'Bob Martinez', 'Carol Williams'],
        'priority_tier': ['High', 'Medium', 'Critical'],
        'sla_target_ms': [100, 150, 250],
        'team': ['Identity', 'Platform', 'Payments'],
        'cost_center': ['CC-1001', 'CC-1002', 'CC-1003']
    })
    
    # Create Excel writer with xlsxwriter engine for formulas
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # =====================================================================
        # Sheet 1: Raw Metrics Data
        # =====================================================================
        df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
        metrics_sheet = writer.sheets['Metrics']
        
        # Format headers
        header_format = workbook.add_format({
            'bold': True, 
            'bg_color': '#2C3E50', 
            'font_color': 'white',
            'border': 1
        })
        for col_num, value in enumerate(df_metrics.columns.values):
            metrics_sheet.write(0, col_num, value, header_format)
        
        # Auto-fit columns
        for i, col in enumerate(df_metrics.columns):
            max_len = max(df_metrics[col].astype(str).map(len).max(), len(col)) + 2
            metrics_sheet.set_column(i, i, max_len)
        
        # =====================================================================
        # Sheet 2: Service Lookup Table
        # =====================================================================
        df_lookup.to_excel(writer, sheet_name='ServiceLookup', index=False)
        lookup_sheet = writer.sheets['ServiceLookup']
        
        for col_num, value in enumerate(df_lookup.columns.values):
            lookup_sheet.write(0, col_num, value, header_format)
        
        for i, col in enumerate(df_lookup.columns):
            max_len = max(df_lookup[col].astype(str).map(len).max(), len(col)) + 2
            lookup_sheet.set_column(i, i, max_len)
        
        # =====================================================================
        # Sheet 3: Business Review with Formulas
        # =====================================================================
        # Create a copy of metrics for business review
        df_review = df_metrics.copy()
        df_review.to_excel(writer, sheet_name='BusinessReview', index=False, startcol=0)
        review_sheet = writer.sheets['BusinessReview']
        
        # Get number of data rows
        num_rows = len(df_review)
        start_col = len(df_review.columns)
        
        # Add formula columns with headers
        formula_headers = ['Owner', 'Priority', 'SLA Target (ms)', 'SLA Status', 'Action Required']
        for i, header in enumerate(formula_headers):
            review_sheet.write(0, start_col + i, header, header_format)
        
        # Add formulas for each row
        # Assuming 'service' is in column A (index 0)
        service_col = 'A'  # Adjust if service column is different
        
        # Check which column has service data
        if 'service' in df_review.columns:
            service_col_idx = df_review.columns.get_loc('service')
            service_col = chr(65 + service_col_idx)  # Convert to letter
        
        # Define formats
        green_format = workbook.add_format({'bg_color': '#27AE60', 'font_color': 'white'})
        yellow_format = workbook.add_format({'bg_color': '#F39C12', 'font_color': 'white'})
        red_format = workbook.add_format({'bg_color': '#E74C3C', 'font_color': 'white'})
        
        for row in range(2, num_rows + 2):  # Excel rows are 1-indexed, plus header
            # VLOOKUP for Owner
            review_sheet.write_formula(
                row - 1, start_col,
                f'=IFERROR(VLOOKUP({service_col}{row},ServiceLookup!$A$2:$F$10,2,FALSE),"Unknown")'
            )
            
            # VLOOKUP for Priority
            review_sheet.write_formula(
                row - 1, start_col + 1,
                f'=IFERROR(VLOOKUP({service_col}{row},ServiceLookup!$A$2:$F$10,3,FALSE),"N/A")'
            )
            
            # VLOOKUP for SLA Target
            review_sheet.write_formula(
                row - 1, start_col + 2,
                f'=IFERROR(VLOOKUP({service_col}{row},ServiceLookup!$A$2:$F$10,4,FALSE),999)'
            )
            
            # IF formula for SLA Status (comparing mean_ms to SLA target)
            # Assuming mean_ms is in column C (index 2)
            mean_col = 'C'  # Adjust based on actual column
            if 'mean_ms' in df_review.columns:
                mean_col_idx = df_review.columns.get_loc('mean_ms')
                mean_col = chr(65 + mean_col_idx)
            
            sla_target_col = chr(65 + start_col + 2)
            review_sheet.write_formula(
                row - 1, start_col + 3,
                f'=IF({mean_col}{row}>{sla_target_col}{row},"SLA BREACH",IF({mean_col}{row}>{sla_target_col}{row}*0.8,"WARNING","OK"))'
            )
            
            # IF formula for Action Required
            review_sheet.write_formula(
                row - 1, start_col + 4,
                f'=IF({mean_col}{row}>{sla_target_col}{row},"Immediate Review",IF({mean_col}{row}>{sla_target_col}{row}*0.8,"Monitor Closely","No Action"))'
            )
        
        # Add conditional formatting for SLA Status column
        sla_status_col = chr(65 + start_col + 3)
        review_sheet.conditional_format(f'{sla_status_col}2:{sla_status_col}{num_rows + 1}', {
            'type': 'text',
            'criteria': 'containing',
            'value': 'BREACH',
            'format': red_format
        })
        review_sheet.conditional_format(f'{sla_status_col}2:{sla_status_col}{num_rows + 1}', {
            'type': 'text',
            'criteria': 'containing',
            'value': 'WARNING',
            'format': yellow_format
        })
        review_sheet.conditional_format(f'{sla_status_col}2:{sla_status_col}{num_rows + 1}', {
            'type': 'text',
            'criteria': 'containing',
            'value': 'OK',
            'format': green_format
        })
        
        # Auto-fit all columns in BusinessReview
        for i in range(start_col + len(formula_headers)):
            review_sheet.set_column(i, i, 15)
        
        # =====================================================================
        # Sheet 4: Summary Dashboard
        # =====================================================================
        summary_sheet = workbook.add_worksheet('Summary')
        
        # Title
        title_format = workbook.add_format({
            'bold': True, 
            'font_size': 16, 
            'bg_color': '#34495E',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        summary_sheet.merge_range('A1:E1', 'OPERATIONAL SYSTEM ANALYTICS - EXECUTIVE SUMMARY', title_format)
        
        # Metrics summary
        metric_label_format = workbook.add_format({'bold': True, 'bg_color': '#ECF0F1'})
        metric_value_format = workbook.add_format({'num_format': '#,##0', 'align': 'right'})
        
        summary_data = [
            ('Total Endpoints Analyzed', len(df_review)),
            ('Services Covered', df_review['service'].nunique() if 'service' in df_review.columns else 'N/A'),
            ('Report Generated', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')),
        ]
        
        for i, (label, value) in enumerate(summary_data):
            summary_sheet.write(i + 2, 0, label, metric_label_format)
            summary_sheet.write(i + 2, 1, value, metric_value_format)
        
        # Add note about formulas
        note_format = workbook.add_format({'italic': True, 'text_wrap': True})
        summary_sheet.write(7, 0, 'Note: BusinessReview sheet contains live formulas (VLOOKUP, IF) that update automatically when source data changes.', note_format)
        summary_sheet.set_column(0, 0, 30)
        summary_sheet.set_column(1, 1, 20)
    
    print(f"✅ Exported business review workbook to: {excel_path}")
    print(f"   📊 Sheets created: Metrics, ServiceLookup, BusinessReview, Summary")
    print(f"   📝 Formulas added: VLOOKUP, IF, IFERROR with conditional formatting")


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    metrics_path = os.path.join(script_dir, '../visualizations/tables/endpoint_performance.csv')
    excel_path = os.path.join(script_dir, '../visualizations/business_review.xlsx')
    
    # Check if metrics file exists
    if not os.path.exists(metrics_path):
        print(f"❌ Metrics file not found: {metrics_path}")
        print("   Run 'python scripts/analysis.py' first to generate metrics.")
    else:
        export_sql_metrics_to_excel(metrics_path, excel_path)
