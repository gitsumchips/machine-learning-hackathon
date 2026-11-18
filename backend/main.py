
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import io

app = FastAPI(
    title="PathSnap API",
    description="GPS Trajectory Map Matching Backend",
    version="1.0.0"
)


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between GPS points in metres."""

    earth_radius = 6371000

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * np.arcsin(np.sqrt(a))


def preprocess_trajectory(df):

    # Remove exact duplicate records
    df = df.drop_duplicates().copy()

    # Identify an available timestamp column
    timestamp_col = next(
        (
            col for col in ["timestamp", "datetime", "time", "timestamp_s"]
            if col in df.columns
        ),
        None
    )

    # Sort by timestamp if available
    if timestamp_col:
        if timestamp_col == "timestamp_s":
            df[timestamp_col] = pd.to_numeric(
                df[timestamp_col], errors="coerce"
            )
        else:
            df[timestamp_col] = pd.to_datetime(
                df[timestamp_col], errors="coerce", utc=True
            )

        df = df.dropna(subset=[timestamp_col])
        df = df.sort_values(timestamp_col)

    df = df.reset_index(drop=True)

    # Calculate distance between consecutive points
    df["distance_from_previous_m"] = 0.0

    if len(df) > 1:
        distances = haversine(
            df["latitude"].iloc[:-1].to_numpy(),
            df["longitude"].iloc[:-1].to_numpy(),
            df["latitude"].iloc[1:].to_numpy(),
            df["longitude"].iloc[1:].to_numpy()
        )

        df.loc[1:, "distance_from_previous_m"] = distances

    # Calculate speed when timestamps exist
    if timestamp_col:
        if timestamp_col == "timestamp_s":
            time_values = df[timestamp_col].to_numpy(dtype=float)
        else:
            time_values = (
                df[timestamp_col].astype("int64").to_numpy()
                / 1_000_000_000
            )

        time_diff = np.diff(time_values, prepend=time_values[0])

        df["speed_mps"] = np.divide(
            df["distance_from_previous_m"],
            time_diff,
            out=np.zeros(len(df), dtype=float),
            where=time_diff > 0
        )

    return df


@app.get("/")
def home():
    return {
        "message": "Welcome to PathSnap API!",
        "status": "Backend is running successfully"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/upload-gps")
async def upload_gps(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file"
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty"
            )

        required_columns = ["latitude", "longitude"]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_columns}"
            )

        original_count = len(df)

        df["latitude"] = pd.to_numeric(
            df["latitude"], errors="coerce"
        )
        df["longitude"] = pd.to_numeric(
            df["longitude"], errors="coerce"
        )

        df = df.dropna(subset=["latitude", "longitude"])

        df = df[
            df["latitude"].between(-90, 90)
            & df["longitude"].between(-180, 180)
        ]

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid GPS coordinates found"
            )

        valid_count = len(df)

        df = preprocess_trajectory(df)

        total_distance = float(
            df["distance_from_previous_m"].sum()
        )

        return {
            "message": "GPS trajectory processed successfully!",
            "filename": file.filename,
            "original_records": original_count,
            "valid_records": valid_count,
            "processed_records": len(df),
            "removed_records": original_count - len(df),
            "total_distance_m": round(total_distance, 2),
            "total_distance_km": round(total_distance / 1000, 4),
            "timestamp_available": (
                "speed_mps" in df.columns
            ),
            "processed_preview": df.head(10).to_dict(
                orient="records"
            )
        }

    except pd.errors.ParserError:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV format"
        )
