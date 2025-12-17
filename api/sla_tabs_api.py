from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
import os
from datetime import datetime
import numpy as np
import json
from typing import List, Dict, Any


sla_tabs_router = APIRouter()


def safe_convert_to_json(df):
    """Convert DataFrame to JSON-safe format handling NaN and inf values"""
    try:
        # Replace NaN and inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(None)
        
        # Convert to records
        records = df.to_dict(orient='records')
        
        # Clean up any remaining problematic values
        cleaned_records = []
        for record in records:
            cleaned_record = {}
            for key, value in record.items():
                if pd.isna(value) or value is None:
                    cleaned_record[key] = None
                elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    cleaned_record[key] = None
                else:
                    cleaned_record[key] = value
            cleaned_records.append(cleaned_record)
        
        return cleaned_records
    except Exception as e:
        print(f"Error converting to JSON: {e}")
        return []


def parse_year_month(date_str):
    """Parse date string to YYYY-MM format"""
    try:
        if pd.isna(date_str):
            return None
        
        # Handle different date formats
        date_str = str(date_str).strip()
        
        # Try different parsing strategies
        formats = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d']
        
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%Y-%m')
            except ValueError:
                continue
        
        # If none of the formats work, return None
        print(f"Could not parse date: {date_str}")
        return None
        
    except Exception as e:
        print(f"Error parsing date {date_str}: {e}")
        return None


def calculate_suspended_statistics(df):
    """
    Calculate suspended statistics by priority and month
    """
    try:
        df = df.copy()
        
        print(f"Calculating suspended statistics for dataset shape: {df.shape}")
        
        # Map actual columns from the CSV
        column_mappings = {
            'req_creation_date': 'Req. Creation Date',
            'priority': 'Request - Priority Description',
            'req_cr_ym': 'ReqCrYM',
            'request_id': 'Request - ID',
            'historical_status': 'Historical Status - Status From',
            'refined_predt': 'RefinedPreDt',
            'current_status': 'Req. Status - Description'
        }
        
        # Verify columns exist
        missing_cols = []
        for key, col_name in column_mappings.items():
            if col_name not in df.columns:
                missing_cols.append(col_name)
        
        if missing_cols:
            print(f"Missing columns for suspended stats: {missing_cols}")
            raise ValueError(f"Critical columns missing from dataset: {missing_cols}")
        
        # Filter for suspended-related statuses
        df_suspended = df[
            (df[column_mappings['historical_status']].str.contains('Suspended|Work in progress', case=False, na=False)) |
            (df[column_mappings['current_status']].str.contains('Suspended|Work in progress', case=False, na=False))
        ].copy()
        
        print(f"Filtered suspended data shape: {df_suspended.shape}")
        
        # Parse dates and create year-month columns (using M/D/Y format)
        df_suspended['req_creation_parsed'] = pd.to_datetime(df_suspended[column_mappings['req_creation_date']], format='%m/%d/%Y', errors='coerce')
        df_suspended['req_cr_ym_parsed'] = df_suspended['req_creation_parsed'].dt.strftime('%Y-%m')
        
        # Use existing ReqCrYM if available, otherwise use parsed dates
        if column_mappings['req_cr_ym'] in df_suspended.columns:
            df_suspended['month_key'] = df_suspended[column_mappings['req_cr_ym']].astype(str).str.strip()
        else:
            df_suspended['month_key'] = df_suspended['req_cr_ym_parsed']
        
        # Clean priority descriptions
        df_suspended['priority_clean'] = df_suspended[column_mappings['priority']].astype(str).fillna('Unknown').str.strip()
        
        # Convert RefinedPreDt to numeric
        df_suspended['refined_predt_numeric'] = pd.to_numeric(df_suspended[column_mappings['refined_predt']], errors='coerce').fillna(0)
        
        # Group by priority and month
        results = []
        grouped = df_suspended.groupby(['priority_clean', 'month_key'])
        
        for (priority, month_key), group_data in grouped:
            try:
                if not month_key or pd.isna(month_key):
                    continue
                    
                count_request_id = len(group_data[column_mappings['request_id']].unique())
                count_historical_status = len(group_data[column_mappings['historical_status']].dropna())
                sum_refined_predt = group_data['refined_predt_numeric'].sum()
                
                if count_request_id > 0:  # Only include if there are requests
                    result = {
                        'priority': priority,
                        'reqCrYM': month_key,
                        'countRequestId': count_request_id,
                        'countHistoricalStatus': count_historical_status,
                        'sumRefinedPreDt': sum_refined_predt
                    }
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing suspended stats for {priority}, {month_key}: {e}")
                continue
        
        # Sort results by priority and month
        results.sort(key=lambda x: (x['priority'], x['reqCrYM']))
        
        print(f"Successfully calculated suspended statistics for {len(results)} records")
        return results
        
    except Exception as e:
        print(f"Error in calculate_suspended_statistics: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_suspended_l2_details(df):
    """
    Calculate detailed suspended L2 ticket information
    """
    try:
        df = df.copy()
        
        print(f"Calculating suspended L2 details for dataset shape: {df.shape}")
        
        # Map actual columns from the CSV
        column_mappings = {
            'req_creation_date': 'Req. Creation Date',
            'priority': 'Request - Priority Description',
            'req_cr_ym': 'ReqCrYM',
            'request_id': 'Request - ID',
            'historical_status_from': 'Historical Status - Status From',
            'historical_status_to': 'Historical Status - Status To',
            'refined_predt': 'RefinedPreDt',
            'consultant': 'Request - Resource Assigned To - Name',
            'current_status': 'Req. Status - Description'
        }
        
        # Verify columns exist
        missing_cols = []
        for key, col_name in column_mappings.items():
            if col_name not in df.columns:
                missing_cols.append(col_name)
        
        if missing_cols:
            print(f"Missing columns for suspended L2: {missing_cols}")
            raise ValueError(f"Critical columns missing from dataset: {missing_cols}")
        
        # Filter for suspended-related statuses
        df_suspended = df[
            (df[column_mappings['historical_status_from']].str.contains('Work in progress|Suspended', case=False, na=False)) |
            (df[column_mappings['historical_status_to']].str.contains('Work in progress|Suspended', case=False, na=False)) |
            (df[column_mappings['current_status']].str.contains('Work in progress|Suspended', case=False, na=False))
        ].copy()
        
        print(f"Filtered suspended L2 data shape: {df_suspended.shape}")
        
        # Parse dates and create year-month columns (using M/D/Y format)
        df_suspended['req_creation_parsed'] = pd.to_datetime(df_suspended[column_mappings['req_creation_date']], format='%m/%d/%Y', errors='coerce')
        df_suspended['req_cr_ym_parsed'] = df_suspended['req_creation_parsed'].dt.strftime('%Y-%m')
        
        # Use existing ReqCrYM if available, otherwise use parsed dates
        if column_mappings['req_cr_ym'] in df_suspended.columns:
            df_suspended['month_key'] = df_suspended[column_mappings['req_cr_ym']].astype(str).str.strip()
        else:
            df_suspended['month_key'] = df_suspended['req_cr_ym_parsed']
        
        # Clean data
        df_suspended['priority_clean'] = df_suspended[column_mappings['priority']].astype(str).fillna('Unknown').str.strip()
        df_suspended['consultant_clean'] = df_suspended[column_mappings['consultant']].astype(str).fillna('Unassigned').str.strip()
        df_suspended['refined_predt_numeric'] = pd.to_numeric(df_suspended[column_mappings['refined_predt']], errors='coerce').fillna(0)
        
        # Get unique consultants for filter section
        consultants = sorted(df_suspended['consultant_clean'].unique())
        consultants = [c for c in consultants if c != 'Unassigned' and c != 'Unknown']
        
        # Group by request ID to get last status for each ticket
        results = []
        grouped = df_suspended.groupby(column_mappings['request_id'])
        
        for request_id, group_data in grouped:
            try:
                # Get the most recent record for this request
                latest_record = group_data.iloc[-1]  # Assuming last row is most recent
                
                # Count historical status records for this request
                count_historical_status = len(group_data[column_mappings['historical_status_from']].dropna())
                
                result = {
                    'reqCrYM': latest_record['month_key'],
                    'requestId': request_id,
                    'priority': latest_record['priority_clean'],
                    'countHistoricalStatus': count_historical_status,
                    'sumRefinedPreDt': group_data['refined_predt_numeric'].sum(),
                    'lastStatusFrom': latest_record[column_mappings['historical_status_from']] if pd.notna(latest_record[column_mappings['historical_status_from']]) else 'Work in progress',
                    'lastStatusTo': latest_record[column_mappings['historical_status_to']] if pd.notna(latest_record[column_mappings['historical_status_to']]) else 'Work in progress',
                    'consultant': latest_record['consultant_clean']
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error processing suspended L2 for request {request_id}: {e}")
                continue
        
        # Sort results by month and request ID
        results.sort(key=lambda x: (x['reqCrYM'], x['requestId']))
        
        print(f"Successfully calculated suspended L2 details for {len(results)} records")
        return results, consultants
        
    except Exception as e:
        print(f"Error in calculate_suspended_l2_details: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_ticket_details(df):
    """
    Calculate detailed ticket information with status history
    """
    try:
        df = df.copy()
        
        print(f"Calculating ticket details for dataset shape: {df.shape}")
        
        # Map actual columns from the CSV
        column_mappings = {
            'req_creation_date': 'Req. Creation Date',
            'priority': 'Request - Priority Description',
            'req_cr_ym': 'ReqCrYM',
            'request_id': 'Request - ID',
            'historical_status_from': 'Historical Status - Status From',
            'historical_status_to': 'Historical Status - Status To',
            'refined_predt': 'RefinedPreDt',
            'consultant': 'Request - Resource Assigned To - Name'
        }
        
        # Verify columns exist
        missing_cols = []
        for key, col_name in column_mappings.items():
            if col_name not in df.columns:
                missing_cols.append(col_name)
        
        if missing_cols:
            print(f"Missing columns for ticket details: {missing_cols}")
            raise ValueError(f"Critical columns missing from dataset: {missing_cols}")
        
        # Parse dates and create year-month columns (M/D/YYYY format)
        df['req_creation_parsed'] = pd.to_datetime(df[column_mappings['req_creation_date']], format='%m/%d/%Y', errors='coerce')
        df['req_cr_ym_parsed'] = df['req_creation_parsed'].dt.strftime('%Y-%m')
        
        # Use existing ReqCrYM if available
        if column_mappings['req_cr_ym'] in df.columns:
            df['month_key'] = df[column_mappings['req_cr_ym']].astype(str).str.strip()
        else:
            df['month_key'] = df['req_cr_ym_parsed']
        
        # Clean data
        df['priority_clean'] = df[column_mappings['priority']].astype(str).fillna('Unknown').str.strip()
        df['consultant_clean'] = df[column_mappings['consultant']].astype(str).fillna('Unassigned').str.strip()
        df['refined_predt_numeric'] = pd.to_numeric(df[column_mappings['refined_predt']], errors='coerce').fillna(0)
        
        results = []
        
        # Get a sample of records (limit to avoid too much data)
        sample_df = df.head(1000)  # Limit to first 1000 records for performance
        
        for index, row in sample_df.iterrows():
            result = {
                'reqCrYM': row['month_key'],
                'requestId': row[column_mappings['request_id']],
                'priority': row['priority_clean'],
                'sumRefinedPreDt': row['refined_predt_numeric'],
                'statusFrom': row[column_mappings['historical_status_from']] if pd.notna(row[column_mappings['historical_status_from']]) else 'N/A',
                'statusTo': row[column_mappings['historical_status_to']] if pd.notna(row[column_mappings['historical_status_to']]) else 'N/A',
                'consultant': row['consultant_clean']
            }
            results.append(result)
        
        print(f"Successfully calculated ticket details for {len(results)} records")
        return results
        
    except Exception as e:
        print(f"Error in calculate_ticket_details: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_open_tickets(df):
    """
    Calculate open tickets data with macro areas and consultant breakdown
    """
    try:
        df = df.copy()
        
        print(f"Calculating open tickets for dataset shape: {df.shape}")
        
        # Map actual columns from the CSV
        column_mappings = {
            'priority': 'Request - Priority Description',
            'consultant': 'Request - Resource Assigned To - Name',
            'macro_area': 'Macro Area - Name',
            'status': 'Req. Status - Description',
            'historical_status_to': 'Historical Status - Status To',
            'request_id': 'Request - ID'
        }
        
        # Filter for open tickets
        df_open = df[
            (~df[column_mappings['status']].str.contains('Closed|Resolved', case=False, na=False))
        ].copy()
        
        print(f"Filtered open tickets shape: {df_open.shape}")
        
        # Clean data
        df_open['priority_clean'] = df_open[column_mappings['priority']].astype(str).fillna('Unknown').str.strip()
        df_open['consultant_clean'] = df_open[column_mappings['consultant']].astype(str).fillna('Unassigned').str.strip()
        df_open['macro_area_clean'] = df_open[column_mappings['macro_area']].astype(str).fillna('Unknown').str.strip()
        
        # Get macro areas
        macro_areas = sorted(df_open['macro_area_clean'].unique())
        macro_areas = [area for area in macro_areas if area not in ['Unknown', '', 'nan', 'NaN']][:10]  # Limit to 10
        
        # Get status options
        status_options = sorted(df_open[column_mappings['historical_status_to']].dropna().unique())[:10]  # Limit to 10
        
        # Calculate consultant priority breakdown
        consultant_priority_data = []
        grouped = df_open.groupby('consultant_clean')
        
        for consultant, group_data in grouped:
            if consultant in ['Unassigned', 'Unknown', '', 'nan', 'NaN']:
                continue
            
            # Get unique tickets by request ID to avoid duplicates    
            unique_tickets = group_data.drop_duplicates(subset=[column_mappings['request_id']])
            priority_counts = unique_tickets['priority_clean'].value_counts()
            
            p1_critical = 0
            p2_high = 0
            p3_normal = 0
            p4_low = 0
            
            for priority, count in priority_counts.items():
                if 'P1' in priority or 'Critical' in priority:
                    p1_critical += count
                elif 'P2' in priority or 'High' in priority:
                    p2_high += count
                elif 'P3' in priority or 'Normal' in priority:
                    p3_normal += count
                elif 'P4' in priority or 'Low' in priority:
                    p4_low += count
            
            total = p1_critical + p2_high + p3_normal + p4_low
            
            if total > 0:
                result = {
                    'consultant': consultant,
                    'p1_critical': p1_critical,
                    'p2_high': p2_high,
                    'p3_normal': p3_normal,
                    'p4_low': p4_low,
                    'total': total
                }
                consultant_priority_data.append(result)
        
        # Sort by total tickets descending
        consultant_priority_data.sort(key=lambda x: x['total'], reverse=True)
        
        print(f"Successfully calculated open tickets for {len(consultant_priority_data)} consultants")
        return consultant_priority_data, macro_areas, status_options
        
    except Exception as e:
        print(f"Error in calculate_open_tickets: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_aging_data(df):
    """
    Calculate aging data for open tickets
    """
    try:
        df = df.copy()
        
        print(f"Calculating aging data for dataset shape: {df.shape}")
        
        # Map actual columns from the CSV
        column_mappings = {
            'req_creation_date': 'Req. Creation Date',
            'req_cr_ym': 'ReqCrYM',
            'request_id': 'Request - ID',
            'status': 'Req. Status - Description'
        }
        
        # Filter for open tickets
        df_open = df[
            (~df[column_mappings['status']].str.contains('Closed|Resolved', case=False, na=False))
        ].copy()
        
        # Parse dates and create year-month columns (using M/D/Y format)
        df_open['req_creation_parsed'] = pd.to_datetime(df_open[column_mappings['req_creation_date']], format='%m/%d/%Y', errors='coerce')
        df_open['req_cr_ym_parsed'] = df_open['req_creation_parsed'].dt.strftime('%Y-%m')
        
        # Use existing ReqCrYM if available
        if column_mappings['req_cr_ym'] in df_open.columns:
            df_open['month_key'] = df_open[column_mappings['req_cr_ym']].astype(str).str.strip()
        else:
            df_open['month_key'] = df_open['req_cr_ym_parsed']
        
        # Group by month and count requests
        aging_data = []
        grouped = df_open.groupby('month_key')
        
        for month_key, group_data in grouped:
            if not month_key or pd.isna(month_key):
                continue
                
            count = len(group_data[column_mappings['request_id']].unique())
            
            if count > 0:
                result = {
                    'reqCrYM': month_key,
                    'count': count
                }
                aging_data.append(result)
        
        # Sort by month
        aging_data.sort(key=lambda x: x['reqCrYM'])
        
        print(f"Successfully calculated aging data for {len(aging_data)} months")
        return aging_data
        
    except Exception as e:
        print(f"Error in calculate_aging_data: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_aging_l2_details(df):
    """
    Calculate aging L2 details with timestamps
    """
    try:
        df = df.copy()
        
        print(f"Calculating aging L2 details for dataset shape: {df.shape}")
        
        # Map actual columns from the CSV
        column_mappings = {
            'consultant': 'Request - Resource Assigned To - Name',
            'historical_change_time': 'Historical Status - Change Time',
            'req_creation_date': 'Req. Creation Date'
        }
        
        # Parse dates for year/month/day extraction (using M/D/Y format)
        df['req_creation_parsed'] = pd.to_datetime(df[column_mappings['req_creation_date']], format='%m/%d/%Y', errors='coerce')
        df['year'] = df['req_creation_parsed'].dt.year
        df['month'] = df['req_creation_parsed'].dt.strftime('%B')
        df['day'] = df['req_creation_parsed'].dt.day
        
        # Clean data
        df['consultant_clean'] = df[column_mappings['consultant']].astype(str).fillna('Unassigned').str.strip()
        df['change_time_clean'] = df[column_mappings['historical_change_time']] if column_mappings['historical_change_time'] in df.columns else 'N/A'
        
        results = []
        
        # Get a sample of records (limit to avoid too much data)
        sample_df = df.head(300)  # Limit to first 300 records for performance
        
        for index, row in sample_df.iterrows():
            result = {
                'consultant': row['consultant_clean'],
                'year': str(int(row['year'])) if pd.notna(row['year']) else 'N/A',
                'month': row['month'] if pd.notna(row['month']) else 'N/A',
                'day': str(int(row['day'])) if pd.notna(row['day']) else 'N/A',
                'changeTime': str(row['change_time_clean']) if pd.notna(row['change_time_clean']) else 'N/A'
            }
            results.append(result)
        
        print(f"Successfully calculated aging L2 details for {len(results)} records")
        return results
        
    except Exception as e:
        print(f"Error in calculate_aging_l2_details: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_sla_monitor(df):
    """
    Calculate SLA monitor data with resolution remaining time
    """
    try:
        df = df.copy()
        
        print(f"Calculating SLA monitor data for dataset shape: {df.shape}")
        
        # Map actual columns from the CSV
        column_mappings = {
            'consultant': 'Request - Resource Assigned To - Name',
            'request_id': 'Request - ID',
            'resol_rem': 'ResolRem'
        }
        
        # Clean data
        df['consultant_clean'] = df[column_mappings['consultant']].astype(str).fillna('Unassigned').str.strip()
        df['resol_rem_numeric'] = pd.to_numeric(df[column_mappings['resol_rem']], errors='coerce').fillna(0)
        
        results = []
        
        # Get a sample of records (limit to avoid too much data)
        sample_df = df.head(200)  # Limit to first 200 records for performance
        
        for index, row in sample_df.iterrows():
            result = {
                'consultant': row['consultant_clean'],
                'requestId': row[column_mappings['request_id']],
                'resolRem': row['resol_rem_numeric']
            }
            results.append(result)
        
        print(f"Successfully calculated SLA monitor data for {len(results)} records")
        return results
        
    except Exception as e:
        print(f"Error in calculate_sla_monitor: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_consultant_statistics(df):
    """
    Calculate consultant-wise summary statistics with assigned and resolved percentages
    """
    try:
        df = df.copy()
        
        print(f"Calculating consultant statistics for dataset shape: {df.shape}")
        
        # Map actual columns from the CSV
        column_mappings = {
            'req_creation_date': 'Req. Creation Date',
            'priority': 'Request - Priority Description',
            'consultant': 'Request - Resource Assigned To - Name',
            'req_cr_ym': 'ReqCrYM',
            'request_id': 'Request - ID',
            'status': 'Req. Status - Description'
        }
        
        # Verify columns exist
        missing_cols = []
        for key, col_name in column_mappings.items():
            if col_name not in df.columns:
                missing_cols.append(col_name)
        
        if missing_cols:
            print(f"Missing columns for consultant stats: {missing_cols}")
            raise ValueError(f"Critical columns missing from dataset: {missing_cols}")
        
        # Clean consultant names
        df['consultant_clean'] = df[column_mappings['consultant']].astype(str).fillna('Unknown').str.strip()
        
        # Filter out invalid consultant names
        df = df[~df['consultant_clean'].isin(['Unknown', 'Unassigned', '', 'nan', 'NaN'])]
        
        # Get total tickets in system (unique tickets)
        total_system_tickets = len(df.drop_duplicates(subset=[column_mappings['request_id']]))
        
        # Check for resolved tickets (closed status)
        df['is_resolved'] = df[column_mappings['status']].astype(str).str.contains('Closed|Resolved', case=False, na=False)
        
        results = []
        
        # Group by consultant
        consultant_groups = df.groupby('consultant_clean')
        
        for consultant, group_data in consultant_groups:
            try:
                # Get unique tickets assigned to this consultant
                unique_tickets = group_data.drop_duplicates(subset=[column_mappings['request_id']])
                assigned_tickets = len(unique_tickets)
                
                # Get resolved tickets for this consultant
                resolved_tickets = len(unique_tickets[unique_tickets['is_resolved'] == True])
                
                # Calculate percentages
                assigned_percentage = (assigned_tickets / total_system_tickets * 100) if total_system_tickets > 0 else 0
                resolved_percentage = (resolved_tickets / assigned_tickets * 100) if assigned_tickets > 0 else 0
                
                result = {
                    'consultant_name': consultant,
                    'assigned_tickets': assigned_tickets,
                    'resolved_tickets': resolved_tickets,
                    'assigned_percentage': round(assigned_percentage, 2),
                    'resolved_percentage': round(resolved_percentage, 2)
                }
                results.append(result)
                    
            except Exception as e:
                print(f"Error processing consultant data for {consultant}: {e}")
                continue
        
        # Sort results by assigned percentage (descending)
        results.sort(key=lambda x: x['assigned_percentage'], reverse=True)
        
        print(f"Successfully calculated consultant statistics for {len(results)} consultants")
        return results, total_system_tickets
        
    except Exception as e:
        print(f"Error in calculate_consultant_statistics: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_consultant_monthly_data(df, consultant_name):
    """
    Calculate monthly created and resolved tickets data for a specific consultant
    """
    try:
        df = df.copy()
        
        print(f"Calculating monthly data for consultant: {consultant_name}")
        
        # Map actual columns from the CSV
        column_mappings = {
            'req_creation_date': 'Req. Creation Date',
            'consultant': 'Request - Resource Assigned To - Name',
            'request_id': 'Request - ID',
            'status': 'Req. Status - Description'
        }
        
        # Verify columns exist
        missing_cols = []
        for key, col_name in column_mappings.items():
            if col_name not in df.columns:
                missing_cols.append(col_name)
        
        if missing_cols:
            print(f"Missing columns for monthly data: {missing_cols}")
            raise ValueError(f"Critical columns missing from dataset: {missing_cols}")
        
        # Clean consultant names and filter for specific consultant
        df['consultant_clean'] = df[column_mappings['consultant']].astype(str).fillna('Unknown').str.strip()
        consultant_data = df[df['consultant_clean'] == consultant_name]
        
        if consultant_data.empty:
            return []
        
        # Parse dates and create year-month columns
        consultant_data['req_creation_parsed'] = pd.to_datetime(
            consultant_data[column_mappings['req_creation_date']], 
            format='%m/%d/%Y', 
            errors='coerce'
        )
        consultant_data['month_key'] = consultant_data['req_creation_parsed'].dt.strftime('%Y-%m')
        
        # Check for resolved tickets
        consultant_data['is_resolved'] = consultant_data[column_mappings['status']].astype(str).str.contains('Closed|Resolved', case=False, na=False)
        
        # Filter out invalid months
        valid_months = consultant_data['month_key'].dropna()
        valid_months = valid_months[valid_months.str.match(r'^\d{4}-\d{2}$', na=False)]
        all_months = sorted(valid_months.unique())
        
        consultant_data = consultant_data[consultant_data['month_key'].isin(all_months)]
        
        results = []
        
        # Group by month
        monthly_groups = consultant_data.groupby('month_key')
        
        for month_key, group_data in monthly_groups:
            try:
                # Parse month for display
                try:
                    year_str, month_str = month_key.split('-')
                    year = int(year_str)
                    month_num = int(month_str)
                    month_name = datetime(year, month_num, 1).strftime('%B %Y')
                except:
                    month_name = month_key
                
                # Get unique tickets created in this month
                unique_tickets = group_data.drop_duplicates(subset=[column_mappings['request_id']])
                tickets_created = len(unique_tickets)
                
                # Get resolved tickets in this month
                tickets_resolved = len(unique_tickets[unique_tickets['is_resolved'] == True])
                
                result = {
                    'month': month_name,
                    'month_key': month_key,
                    'tickets_created': tickets_created,
                    'tickets_resolved': tickets_resolved
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error processing monthly data for {month_key}: {e}")
                continue
        
        # Sort by month
        results.sort(key=lambda x: x['month_key'])
        
        print(f"Successfully calculated monthly data for {len(results)} months")
        return results
        
    except Exception as e:
        print(f"Error in calculate_consultant_monthly_data: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_tickets_statistics(df):
    """
    Calculate tickets statistics based on actual CSV data structure
    """
    try:
        # Create necessary columns for calculations
        df = df.copy()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        
        # Map actual columns from the CSV
        column_mappings = {
            'req_creation_date': 'Req. Creation Date',
            'resp_sla': 'RespSLA', 
            'resol_sla': 'ResolSLA',
            'req_status': 'Req. Status - Description',
            'rollover': 'Rollover',
            'req_cr_ym': 'ReqCrYM',
            'resp_remaining': 'RespRem',
            'resol_remaining': 'ResolRem'
        }
        
        # Verify columns exist
        missing_cols = []
        for key, col_name in column_mappings.items():
            if col_name not in df.columns:
                missing_cols.append(col_name)
        
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            # If critical columns are missing, raise an error instead of using fallback
            raise ValueError(f"Critical columns missing from dataset: {missing_cols}")
        
        # Parse dates and create year-month columns (M/D/YYYY format)
        df['req_creation_parsed'] = pd.to_datetime(df[column_mappings['req_creation_date']], format='%m/%d/%Y', errors='coerce')
        df['req_cr_ym_parsed'] = df['req_creation_parsed'].dt.strftime('%Y-%m')
        
        # Use only parsed dates since ReqCrYM column contains invalid Excel serial numbers
        df['month_key'] = df['req_cr_ym_parsed']
        
        # Handle rollover dates - ignore the corrupted Rollover column and calculate properly
        # Rollover should be based on ticket status, not a separate column with Excel serial numbers
        df['rollover_ym'] = df['month_key']  # We'll calculate rollover logic differently
        
        # Create boolean columns for SLA calculations
        df['resp_sla_yes'] = df[column_mappings['resp_sla']].astype(str).str.contains('Yes', case=False, na=False)
        df['resol_sla_yes'] = df[column_mappings['resol_sla']].astype(str).str.contains('Yes', case=False, na=False)
        df['req_closed'] = df[column_mappings['req_status']].astype(str).str.contains('Closed', case=False, na=False)
        
        # Handle remaining time columns
        df['resp_rem_numeric'] = pd.to_numeric(df[column_mappings['resp_remaining']], errors='coerce')
        df['resol_rem_numeric'] = pd.to_numeric(df[column_mappings['resol_remaining']], errors='coerce')
        
        df['resp_rem_positive'] = df['resp_rem_numeric'] >= 0
        df['resol_rem_positive'] = df['resol_rem_numeric'] >= 0
        
        # Get unique months for calculation (filter out invalid values)
        valid_months = df['month_key'].dropna()
        valid_months = valid_months[valid_months.str.match(r'^\d{4}-\d{2}$', na=False)]
        all_months = sorted(valid_months.unique())
        
        # Filter data to only include valid months to avoid duplication
        df = df[df['month_key'].isin(all_months)]
        
        print(f"Calculating for months: {all_months[:10]}...")  # Show first 10 months
        
        results = []
        
        for selected_month in all_months:
            try:
                # Parse month for display
                try:
                    # Handle different month formats (YYYY MM, YYYY-MM, etc.)
                    month_clean = selected_month.replace(' ', '-').replace('/', '-')
                    if len(month_clean.split('-')) == 2:
                        year_str, month_str = month_clean.split('-')
                        year = int(year_str)
                        month_num = int(month_str)
                        month_name = datetime(year, month_num, 1).strftime('%B')
                    else:
                        year = selected_month[:4]
                        month_name = selected_month
                except:
                    year = selected_month[:4] if len(selected_month) >= 4 else "2024"
                    month_name = selected_month
                
                # Filter data for selected month
                month_data = df[df['month_key'] == selected_month]
                
                # TicketsCreated - tickets created in selected month with RespSLA = Yes
                tickets_created = len(month_data[month_data['resp_sla_yes'] == True])
                
                # TotalTicketsInclRollover - calculate cumulative active tickets properly
                # Get all tickets created in or before this month
                created_before_or_in_month = df[df['month_key'] <= selected_month]
                
                # Get unique tickets and their latest status for this calculation
                latest_status_by_ticket = created_before_or_in_month.groupby('Request - ID').last()
                
                # Count tickets that are still active (not closed) by this month
                active_tickets = latest_status_by_ticket[~latest_status_by_ticket['req_closed']]
                total_tickets = len(active_tickets)
                
                # TicketsCompleted - tickets closed in selected month
                tickets_completed = len(month_data[month_data['req_closed'] == True])
                
                # ResponseSLAMet - tickets with response SLA met
                response_sla_met = len(month_data[
                    (month_data['resp_sla_yes'] == True) & 
                    (month_data['resp_rem_positive'] == True)
                ])
                
                # Get completed tickets for this month
                completed_tickets_data = month_data[month_data['req_closed'] == True]
                
                # Among completed tickets, count how many had resolution SLA met
                resolution_sla_met = len(completed_tickets_data[
                    (completed_tickets_data['resol_sla_yes'] == True) & 
                    (completed_tickets_data['resol_rem_positive'] == True)
                ])
                
                # Calculate percentages
                response_sla_percent = (response_sla_met / tickets_created * 100) if tickets_created > 0 else 0
                resolution_sla_percent = (resolution_sla_met / tickets_completed * 100) if tickets_completed > 0 else 0
                
                # ModSLAMet % - Combined SLA performance
                mod_sla_met = (response_sla_percent + resolution_sla_percent) / 2
                
                # BothSLAsMet % - Conservative calculation
                both_slas_met = min(response_sla_percent, resolution_sla_percent)
                
                # Only include months with actual data
                if tickets_created > 0 or total_tickets > 0 or tickets_completed > 0:
                    result = {
                        'year': str(year),
                        'month': month_name,
                        'ticketsCreated': tickets_created,
                        'totalTicketsInclRollover': total_tickets,
                        'ticketsCompleted': tickets_completed,
                        'responseSLA': round(response_sla_percent, 2),
                        'resolutionSLA': round(resolution_sla_percent, 2),
                        'modSLAMet': round(mod_sla_met, 2),
                        'bothSLAsMet': round(both_slas_met, 2),
                        'resolutionSLATime': resolution_sla_met,
                        'responseSLAMet': response_sla_met,
                        'resolutionSLAMet': resolution_sla_met
                    }
                    
                    results.append(result)
                
            except Exception as e:
                print(f"Error calculating for month {selected_month}: {e}")
                continue
        
        print(f"Successfully calculated statistics for {len(results)} months")
        return results
        
    except Exception as e:
        print(f"Error in calculate_tickets_statistics: {e}")
        import traceback
        traceback.print_exc()
        # Don't return empty list, raise the error instead
        raise
        

@sla_tabs_router.get("/api/sla_tabs/{tab_name}")
async def get_tab_data(tab_name: str):
    """
    Get data for specific SLA tab
    """
    try:
        # Load data from uploads_sla
        csv_file_path = os.path.join('uploads_sla', 'data1.csv')
        if not os.path.exists(csv_file_path):
            raise HTTPException(status_code=404, detail="No data file found. Please upload a file first.")
        
        df = pd.read_csv(csv_file_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty.")
        
        print(f"Loading tab data for: {tab_name}")
        print(f"Dataset shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        
        if tab_name == "Tkts_SLAs_Table":
            # Calculate ticket statistics dynamically from real data
            statistics = calculate_tickets_statistics(df)
            
            if not statistics:
                raise HTTPException(status_code=500, detail="No statistics could be calculated from the dataset. Please check data quality and column mappings.")
            
            # Prepare chart data separately to ensure it's properly formatted
            chart_data = {
                    "months": [f"{item['year']} {item['month']}" for item in statistics],
                    "ticketsCreated": [item['ticketsCreated'] for item in statistics],
                    "totalTicketsInclRollover": [item['totalTicketsInclRollover'] for item in statistics],
                    "responseSLA": [item['responseSLA'] for item in statistics],
                    "resolutionSLA": [item['resolutionSLA'] for item in statistics],
                    "modSLAMet": [item['modSLAMet'] for item in statistics]
            }
            
            return JSONResponse(content={
                "tab_name": tab_name,
                "data_type": "table_and_chart",
                "table_data": statistics,
                "chart_data": chart_data,
                "chart_config": {
                    "chart_type": "combo",
                    "title": "Tickets - Inflow, Incl. Rollover, SLAs",
                    "x_axis_title": "Month",
                    "y_axis_title": "Ticket Count",
                    "y2_axis_title": "SLA Percentage"
                },
                "summary": {
                    "total_months": len(statistics),
                    "total_tickets_created": sum([item['ticketsCreated'] for item in statistics]),
                    "total_tickets_completed": sum([item['ticketsCompleted'] for item in statistics]),
                    "average_response_sla": round(sum([item['responseSLA'] for item in statistics]) / len(statistics), 2) if statistics else 0,
                    "average_resolution_sla": round(sum([item['resolutionSLA'] for item in statistics]) / len(statistics), 2) if statistics else 0
                },
                "message": f"Successfully calculated statistics for {len(statistics)} months"
            })
        
        elif tab_name == "Tkts_SLAs_Chart":
            # Return chart data for the chart tab
            statistics = calculate_tickets_statistics(df)
            
            if not statistics:
                raise HTTPException(status_code=500, detail="No chart data could be calculated from the dataset.")
                
            return JSONResponse(content={
                "tab_name": tab_name,
                "data_type": "chart_data",
                "chart_data": {
                    "months": [f"{item['year']} {item['month']}" for item in statistics],
                    "ticketsCreated": [item['ticketsCreated'] for item in statistics],
                    "totalTicketsInclRollover": [item['totalTicketsInclRollover'] for item in statistics],
                    "responseSLA": [item['responseSLA'] for item in statistics],
                    "resolutionSLA": [item['resolutionSLA'] for item in statistics]
                },
                "message": f"Successfully calculated chart data for {len(statistics)} months"
            })
            
        elif tab_name == "Consultant_Wise":
            # Return consultant-wise data with assigned and resolved percentages
            consultant_stats, total_system_tickets = calculate_consultant_statistics(df)
            
            if not consultant_stats:
                raise HTTPException(status_code=500, detail="No consultant data could be calculated from the dataset.")
            
            # Calculate summary statistics
            total_consultants = len(consultant_stats)
            total_assigned_tickets = sum(item['assigned_tickets'] for item in consultant_stats)
            total_resolved_tickets = sum(item['resolved_tickets'] for item in consultant_stats)
            avg_assigned_percentage = sum(item['assigned_percentage'] for item in consultant_stats) / total_consultants if total_consultants > 0 else 0
            avg_resolved_percentage = sum(item['resolved_percentage'] for item in consultant_stats) / total_consultants if total_consultants > 0 else 0
            
            return JSONResponse(content={
                "tab_name": tab_name,
                "data_type": "consultant_table",
                "consultant_data": consultant_stats,
                "summary": {
                    "total_consultants": total_consultants,
                    "total_system_tickets": total_system_tickets,
                    "total_assigned_tickets": total_assigned_tickets,
                    "total_resolved_tickets": total_resolved_tickets,
                    "avg_assigned_percentage": round(avg_assigned_percentage, 2),
                    "avg_resolved_percentage": round(avg_resolved_percentage, 2)
                },
                "message": f"Successfully calculated consultant data for {len(consultant_stats)} consultants"
            })
            
        elif tab_name == "Suspended_Stats":
            # Return suspended statistics data
            suspended_stats = calculate_suspended_statistics(df)
            
            if not suspended_stats:
                raise HTTPException(status_code=500, detail="No suspended statistics could be calculated from the dataset.")
            
            # Calculate summary statistics
            total_requests = sum(item['countRequestId'] for item in suspended_stats)
            total_historical_status = sum(item['countHistoricalStatus'] for item in suspended_stats)
            total_refined_predt = sum(item['sumRefinedPreDt'] for item in suspended_stats)
            
            return JSONResponse(content={
                "tab_name": tab_name,
                "data_type": "suspended_stats",
                "suspended_data": suspended_stats,
                "summary": {
                    "total_requests": total_requests,
                    "total_historical_status": total_historical_status,
                    "total_refined_predt": total_refined_predt
                },
                "message": f"Successfully calculated suspended statistics for {len(suspended_stats)} records"
            })
            
        elif tab_name == "Suspended_L2":
            # Return suspended L2 detailed data
            suspended_l2_data, consultants = calculate_suspended_l2_details(df)
            
            if not suspended_l2_data:
                raise HTTPException(status_code=500, detail="No suspended L2 data could be calculated from the dataset.")
            
            # Calculate summary statistics
            total_historical_status = sum(item['countHistoricalStatus'] for item in suspended_l2_data)
            total_refined_predt = sum(item['sumRefinedPreDt'] for item in suspended_l2_data)
            
            return JSONResponse(content={
                "tab_name": tab_name,
                "data_type": "suspended_l2",
                "suspended_l2_data": suspended_l2_data,
                "consultants": consultants,
                "summary": {
                    "total_tickets": len(suspended_l2_data),
                    "total_historical_status": total_historical_status,
                    "total_refined_predt": total_refined_predt
                },
                "message": f"Successfully calculated suspended L2 data for {len(suspended_l2_data)} tickets with {len(consultants)} consultants"
            })
            
        elif tab_name == "Tkt_Details":
            # Return ticket details data
            ticket_details = calculate_ticket_details(df)
            
            if not ticket_details:
                raise HTTPException(status_code=500, detail="No ticket details could be calculated from the dataset.")
            
            # Calculate summary statistics
            total_refined_predt = sum(item['sumRefinedPreDt'] for item in ticket_details)
            
            return JSONResponse(content={
                "tab_name": tab_name,
                "data_type": "ticket_details",
                "ticket_details": ticket_details,
                "summary": {
                    "total_records": len(ticket_details),
                    "total_refined_predt": total_refined_predt
                },
                "message": f"Successfully calculated ticket details for {len(ticket_details)} records"
            })
            
        elif tab_name == "Open_Tkts":
            # Return open tickets data with macro areas
            open_tickets, macro_areas, status_options = calculate_open_tickets(df)
            
            if not open_tickets:
                raise HTTPException(status_code=500, detail="No open tickets data could be calculated from the dataset.")
            
            # Calculate summary statistics
            total_p1 = sum(item['p1_critical'] for item in open_tickets)
            total_p2 = sum(item['p2_high'] for item in open_tickets)
            total_p3 = sum(item['p3_normal'] for item in open_tickets)
            total_p4 = sum(item['p4_low'] for item in open_tickets)
            grand_total = sum(item['total'] for item in open_tickets)
            
            return JSONResponse(content={
                "tab_name": tab_name,
                "data_type": "open_tickets",
                "open_tickets": open_tickets,
                "macro_areas": macro_areas,
                "status_options": status_options,
                "summary": {
                    "total_p1": total_p1,
                    "total_p2": total_p2,
                    "total_p3": total_p3,
                    "total_p4": total_p4,
                    "grand_total": grand_total
                },
                "message": f"Successfully calculated open tickets for {len(open_tickets)} consultants"
            })
            
        elif tab_name == "Open_Tkts_L2":
            # Return aging data for open tickets
            aging_data = calculate_aging_data(df)
            
            if not aging_data:
                raise HTTPException(status_code=500, detail="No aging data could be calculated from the dataset.")
            
            # Prepare chart data
            months = [item['reqCrYM'] for item in aging_data]
            counts = [item['count'] for item in aging_data]
            
            chart_data = {
                "months": months,
                "counts": counts
            }
            
            total_count = sum(counts)
            
            return JSONResponse(content={
                "tab_name": tab_name,
                "data_type": "aging_analysis",
                "aging_data": aging_data,
                "chart_data": chart_data,
                "summary": {
                    "total_count": total_count,
                    "total_months": len(aging_data)
                },
                "message": f"Successfully calculated aging data for {len(aging_data)} months"
            })
            
        elif tab_name == "Aging_L2":
            # Return aging L2 details with timestamps
            aging_l2_details = calculate_aging_l2_details(df)
            
            if not aging_l2_details:
                raise HTTPException(status_code=500, detail="No aging L2 details could be calculated from the dataset.")
            
            # Calculate summary statistics
            unique_consultants = len(set(item['consultant'] for item in aging_l2_details))
            
            return JSONResponse(content={
                "tab_name": tab_name,
                "data_type": "aging_l2_details",
                "aging_l2_details": aging_l2_details,
                "summary": {
                    "total_records": len(aging_l2_details),
                    "unique_consultants": unique_consultants
                },
                "message": f"Successfully calculated aging L2 details for {len(aging_l2_details)} records"
            })
            
        elif tab_name == "SLA_Monitor":
            # Return SLA monitor data
            sla_data = calculate_sla_monitor(df)
            
            if not sla_data:
                raise HTTPException(status_code=500, detail="No SLA monitor data could be calculated from the dataset.")
            
            # Calculate summary statistics
            overdue_count = sum(1 for item in sla_data if item['resolRem'] < 0)
            critical_count = sum(1 for item in sla_data if 0 <= item['resolRem'] < 24)
            warning_count = sum(1 for item in sla_data if 24 <= item['resolRem'] < 72)
            safe_count = sum(1 for item in sla_data if item['resolRem'] >= 72)
            avg_resol_rem = sum(item['resolRem'] for item in sla_data) / len(sla_data) if sla_data else 0
            
            return JSONResponse(content={
                "tab_name": tab_name,
                "data_type": "sla_monitor",
                "sla_data": sla_data,
                "summary": {
                    "total_records": len(sla_data),
                    "overdue_count": overdue_count,
                    "critical_count": critical_count,
                    "warning_count": warning_count,
                    "safe_count": safe_count,
                    "avg_resol_rem": avg_resol_rem
                },
                "message": f"Successfully calculated SLA monitor data for {len(sla_data)} records"
            })
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tab name: {tab_name}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_tab_data: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@sla_tabs_router.get("/api/sla_tabs/chart_data/{tab_name}")
async def get_chart_data(tab_name: str):
    """
    Get chart data specifically for a tab
    """
    try:
        csv_file_path = os.path.join('uploads_sla', 'data1.csv')
        if not os.path.exists(csv_file_path):
            raise HTTPException(status_code=404, detail="No data file found.")
        
        df = pd.read_csv(csv_file_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty.")
        
        if tab_name == "Tkts_SLAs_Table":
            statistics = calculate_tickets_statistics(df)
            
            if not statistics:
                raise HTTPException(status_code=500, detail="No chart data could be generated from the dataset.")
            
            return JSONResponse(content={
                "chart_type": "combo",
                "title": "Tickets - Inflow, Incl. Rollover, SLAs",
                "months": [f"{item['year']} {item['month']}" for item in statistics],
                "series": [
                    {
                        "name": "TicketsCreated",
                        "type": "bar",
                        "data": [item['ticketsCreated'] for item in statistics],
                        "color": "#3498db"
                    },
                    {
                        "name": "TotalTicketsInclRollover",
                        "type": "bar", 
                        "data": [item['totalTicketsInclRollover'] for item in statistics],
                        "color": "#2c3e50"
                    },
                    {
                        "name": "ResponseSLA %",
                        "type": "line",
                        "data": [item['responseSLA'] for item in statistics],
                        "color": "#e67e22"
                    },
                    {
                        "name": "ResolutionSLA %",
                        "type": "line",
                        "data": [item['resolutionSLA'] for item in statistics],
                        "color": "#8e44ad"
                    }
                ],
                "message": f"Chart data calculated for {len(statistics)} months"
            })
        elif tab_name == "Tkts_SLAs_Chart":
            statistics = calculate_tickets_statistics(df)
            
            if not statistics:
                raise HTTPException(status_code=500, detail="No chart data could be generated from the dataset.")
            
            return JSONResponse(content={
                "chart_type": "combo",
                "title": "Tickets - Inflow, Incl. Rollover, SLAs",
                "months": [f"{item['year']} {item['month']}" for item in statistics],
                "series": [
                    {
                        "name": "TicketsCreated",
                        "type": "bar",
                        "data": [item['ticketsCreated'] for item in statistics],
                        "color": "#3498db"
                    },
                    {
                        "name": "TotalTicketsInclRollover",
                        "type": "bar", 
                        "data": [item['totalTicketsInclRollover'] for item in statistics],
                        "color": "#2c3e50"
                    },
                    {
                        "name": "ResponseSLA %",
                        "type": "line",
                        "data": [item['responseSLA'] for item in statistics],
                        "color": "#e67e22"
                    },
                    {
                        "name": "ResolutionSLA %",
                        "type": "line",
                        "data": [item['resolutionSLA'] for item in statistics],
                        "color": "#8e44ad"
                    }
                ],
                "message": f"Chart data calculated for {len(statistics)} months"
            })
        else:
            return JSONResponse(content={
                "message": "Chart data not available for this tab",
                "chart_type": "none"
            })
            
    except Exception as e:
        print(f"Error getting chart data: {e}")
        raise HTTPException(status_code=500, detail=f"Chart data error: {str(e)}")


@sla_tabs_router.get("/api/sla_tabs/consultant_monthly/{consultant_name}")
async def get_consultant_monthly_data(consultant_name: str):
    """
    Get monthly created and resolved tickets data for a specific consultant
    """
    try:
        # Load data from uploads_sla
        csv_file_path = os.path.join('uploads_sla', 'data1.csv')
        if not os.path.exists(csv_file_path):
            raise HTTPException(status_code=404, detail="No data file found. Please upload a file first.")
        
        df = pd.read_csv(csv_file_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty.")
        
        print(f"Loading monthly data for consultant: {consultant_name}")
        
        # Calculate monthly data for the specific consultant
        monthly_data = calculate_consultant_monthly_data(df, consultant_name)
        
        if not monthly_data:
            raise HTTPException(status_code=404, detail=f"No data found for consultant: {consultant_name}")
        
        # Prepare chart data
        chart_data = {
            "months": [item['month'] for item in monthly_data],
            "tickets_created": [item['tickets_created'] for item in monthly_data],
            "tickets_resolved": [item['tickets_resolved'] for item in monthly_data]
        }
        
        # Calculate summary
        total_created = sum(item['tickets_created'] for item in monthly_data)
        total_resolved = sum(item['tickets_resolved'] for item in monthly_data)
        
        return JSONResponse(content={
            "consultant_name": consultant_name,
            "monthly_data": monthly_data,
            "chart_data": chart_data,
            "summary": {
                "total_months": len(monthly_data),
                "total_created": total_created,
                "total_resolved": total_resolved,
                "resolution_rate": round((total_resolved / total_created * 100), 2) if total_created > 0 else 0
            },
            "message": f"Successfully calculated monthly data for {consultant_name} across {len(monthly_data)} months"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_consultant_monthly_data: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
