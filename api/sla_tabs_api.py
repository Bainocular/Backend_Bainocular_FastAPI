from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
from datetime import datetime
import pandas as pd


sla_tabs_router = APIRouter()


def calculate_tickets_statistics(df: pd.DataFrame):
    """
    Compute per-month metrics needed by Tkts_SLAs_Chart using expected CSV columns.
    Required columns:
      - 'Req. Creation Date' (M/D/YYYY)
      - 'RespSLA' ('Yes' when applicable)
      - 'ResolSLA' ('Yes' when applicable)
      - 'Req. Status - Description' (contains 'Closed' when closed)
      - 'RespRem' numeric (>=0 means met)
      - 'ResolRem' numeric (>=0 means met)
      - 'Request - ID' ticket id
    """
    # Map CSV columns
    cols = {
        "creation": "Req. Creation Date",
        "resp_sla": "RespSLA",
        "resol_sla": "ResolSLA",
        "status": "Req. Status - Description",
        "resp_rem": "RespRem",
        "resol_rem": "ResolRem",
        "ticket_id": "Request - ID",
    }
    missing = [v for v in cols.values() if v not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse creation date -> YYYY-MM key
        df = df.copy()
    df["_created_dt"] = pd.to_datetime(df[cols["creation"]], format="%m/%d/%Y", errors="coerce")
    df["_month_key"] = df["_created_dt"].dt.strftime("%Y-%m")

    # Normalize flags
    df["_resp_yes"] = df[cols["resp_sla"]].astype(str).str.contains("yes", case=False, na=False)
    df["_resol_yes"] = df[cols["resol_sla"]].astype(str).str.contains("yes", case=False, na=False)
    df["_closed"] = df[cols["status"]].astype(str).str.contains("closed", case=False, na=False)
    df["_resp_rem_pos"] = pd.to_numeric(df[cols["resp_rem"]], errors="coerce").fillna(-1) >= 0
    df["_resol_rem_pos"] = pd.to_numeric(df[cols["resol_rem"]], errors="coerce").fillna(-1) >= 0

    valid_months = sorted(m for m in df["_month_key"].dropna().unique())
        results = []
    for month_key in valid_months:
        month_df = df[df["_month_key"] == month_key]
        # Tickets created in the month with RespSLA = Yes
        tickets_created = int((month_df["_resp_yes"]).sum())

        # TotalTicketsInclRollover: count tickets still active by this month
        created_upto_month = df[df["_month_key"] <= month_key]
        latest_status = created_upto_month.sort_values(["_month_key"]).groupby(cols["ticket_id"]).tail(1)
        total_tickets_incl_rollover = int((~latest_status["_closed"]).sum())

        # Tickets completed (closed) in this month
        tickets_completed = int((month_df["_closed"]).sum())

        # SLA percentages
        response_sla_met = int(((month_df["_resp_yes"]) & (month_df["_resp_rem_pos"])).sum())
        resolution_sla_met = int(((month_df["_resol_yes"]) & (month_df["_resol_rem_pos"]) & (month_df["_closed"])).sum())

        response_sla_pct = (response_sla_met / tickets_created * 100.0) if tickets_created > 0 else 0.0
        resolution_sla_pct = (resolution_sla_met / tickets_completed * 100.0) if tickets_completed > 0 else 0.0

        # Display month label (e.g., "2024 January")
        try:
            year, month = month_key.split("-")
            month_name = datetime(int(year), int(month), 1).strftime("%B")
        except Exception:
            year = month_key[:4]
            month_name = month_key

        results.append(
            {
                "year": str(year),
                "month": month_name,
                "ticketsCreated": tickets_created,
                "totalTicketsInclRollover": total_tickets_incl_rollover,
                "ticketsCompleted": tickets_completed,
                "responseSLA": round(response_sla_pct, 2),
                "resolutionSLA": round(resolution_sla_pct, 2),
            }
        )

        return results
        

@sla_tabs_router.get("/api/sla_tabs/{tab_name}")
async def get_tab_data(tab_name: str):
    # Only support the requested chart endpoint
    if tab_name != "Tkts_SLAs_Chart":
        raise HTTPException(status_code=404, detail="Tab not available")

    csv_path = os.path.join("uploads_sla", "data1.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="No data file found. Please upload a file first.")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    try:
        stats = calculate_tickets_statistics(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute statistics: {e}")

    return JSONResponse(
        content={
            "tab_name": tab_name,
            "data_type": "chart_data",
            "chart_data": {
                "months": [f"{item['year']} {item['month']}" for item in stats],
                "ticketsCreated": [item["ticketsCreated"] for item in stats],
                "totalTicketsInclRollover": [item["totalTicketsInclRollover"] for item in stats],
                "responseSLA": [item["responseSLA"] for item in stats],
                "resolutionSLA": [item["resolutionSLA"] for item in stats],
            },
            "message": f"Successfully calculated chart data for {len(stats)} months",
        }
    )


