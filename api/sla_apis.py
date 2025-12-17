from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
import os
import gzip
import shutil
import time
import pandas as pd


sla_router = APIRouter()


def _ensure_upload_dir() -> str:
    upload_dir = "uploads_sla"
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


@sla_router.post("/api/upload_data")
async def upload_data_only(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    upload_dir = _ensure_upload_dir()

    # .csv.gz → decompress to data1.csv
    if ext == ".gz":
        inner_ext = os.path.splitext(filename[:-3])[1].lower()
        if inner_ext != ".csv":
            raise HTTPException(status_code=400, detail="Only .csv.gz files are supported")

        t_start = time.monotonic()
        target_path = os.path.join(upload_dir, "data1.csv")
        try:
            content = await file.read()
            with gzip.open(io.BytesIO(content), "rb") as gz_file:
                with open(target_path, "wb") as out_f:
                    shutil.copyfileobj(gz_file, out_f, length=1024 * 1024)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to decompress and save CSV.GZ file: {e}")

        # track last processed filename (best-effort)
        try:
            with open(os.path.join(upload_dir, "processed_files.txt"), "w", encoding="utf-8") as f:
                f.write(f"{filename}\n")
        except Exception:
            pass

        total_ms = int((time.monotonic() - t_start) * 1000)
        return JSONResponse(
            content={"message": "Gzipped CSV uploaded and decompressed", "filename": "data1.csv"},
            headers={"Server-Timing": f"decompress_save;dur={total_ms}"},
        )

    # .csv → stream save to data1.csv
    if ext == ".csv":
        t_start = time.monotonic()
        target_path = os.path.join(upload_dir, "data1.csv")
        try:
            try:
                file.file.seek(0)
            except Exception:
                pass
            with open(target_path, "wb") as out_f:
                shutil.copyfileobj(file.file, out_f, length=1024 * 1024)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save CSV file: {e}")

        try:
            with open(os.path.join(upload_dir, "processed_files.txt"), "w", encoding="utf-8") as f:
                f.write(f"{filename}\n")
        except Exception:
            pass

        total_ms = int((time.monotonic() - t_start) * 1000)
        return JSONResponse(
            content={"message": "File uploaded successfully", "filename": "data1.csv"},
            headers={"Server-Timing": f"save;dur={total_ms}"},
        )

    # .xls/.xlsx → convert to CSV as data1.csv
    if ext in [".xls", ".xlsx"]:
        t_start = time.monotonic()
        try:
            content = await file.read()
            df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read Excel file: {e}")

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty file or no data found")

        try:
            df.to_csv(os.path.join(upload_dir, "data1.csv"), index=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save converted CSV: {e}")

        try:
            with open(os.path.join(upload_dir, "processed_files.txt"), "w", encoding="utf-8") as f:
                f.write(f"{filename}\n")
        except Exception:
            pass

        total_ms = int((time.monotonic() - t_start) * 1000)
        return JSONResponse(
            content={"message": "Excel uploaded and converted successfully", "filename": "data1.csv"},
            headers={"Server-Timing": f"excel_read_convert;dur={total_ms}"},
        )

    raise HTTPException(status_code=400, detail="Unsupported file type")


def _convert_to_json_safe(df: pd.DataFrame):
    # Replace inf values and convert NaNs to None for JSON safety
    df = df.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    records = df.to_dict(orient="records")
    cleaned = []
    for rec in records:
        cleaned.append({k: (None if pd.isna(v) else v) for k, v in rec.items()})
    return cleaned


@sla_router.get("/api/get_csv_data/{filename}")
async def get_csv_data(filename: str):
    # sanitize and only allow csv/xlsx/xls
    filename = os.path.basename(filename)
    if not filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = os.path.join("uploads_sla", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    
    if df.empty:
        raise HTTPException(status_code=400, detail="File is empty or contains no data")

    records = _convert_to_json_safe(df)
    return JSONResponse(
        content={"filename": filename, "records": records, "total_rows": len(records), "columns": list(df.columns)}
    )


