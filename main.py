#!/usr/bin/env python3
"""
Cybercast Predictive Analytics API (SIH 2024, Problem Statement 184)
Production FastAPI backend optimized for Railway.app deployment.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cybercast-api")

# -----------------------------------------------------------------------------
# SAFE PATH RESOLUTION (Works on local dev, Docker, and Railway containers)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

def resolve_path(subdir: Path, filename: str) -> Path:
    """Check subdir first, fallback to root BASE_DIR if directory is flattened."""
    target = subdir / filename
    if target.exists():
        return target
    root_target = BASE_DIR / filename
    if root_target.exists():
        return root_target
    return target  # Return expected path even if missing for error reporting

# -----------------------------------------------------------------------------
# GLOBAL MODEL REGISTRY & LIFESPAN
# -----------------------------------------------------------------------------
MODELS: Dict[str, Any] = {
    "loaded": False,
    "zone_stage1": None,
    "zone_stage2_model": None,
    "zone_stage2_classes": None,
    "state_model": None,
    "state_classes": None,
    "label_encoders": None,
    "district_model": None,
    "district_scaler": None,
    "risk_df": None,
    "errors": []
}

def load_ml_assets():
    """Safely loads all v3 models and data tables without crashing on import."""
    errors = []
    
    # 1. Zone Predictor Stage 1
    p_s1 = resolve_path(MODELS_DIR, "zone_predictor_v3_stage1.pkl")
    try:
        if p_s1.exists():
            MODELS["zone_stage1"] = joblib.load(p_s1)
            logger.info(f"Loaded Zone Stage 1: {p_s1.name}")
        else:
            errors.append(f"Missing {p_s1}")
    except Exception as e:
        errors.append(f"Error loading {p_s1.name}: {str(e)}")

    # 2. Zone Predictor Stage 2
    p_s2 = resolve_path(MODELS_DIR, "zone_predictor_v3_stage2.pkl")
    try:
        if p_s2.exists():
            s2_data = joblib.load(p_s2)
            MODELS["zone_stage2_model"] = s2_data["model"]
            MODELS["zone_stage2_classes"] = list(s2_data["classes"])
            logger.info(f"Loaded Zone Stage 2: {p_s2.name}")
        else:
            errors.append(f"Missing {p_s2}")
    except Exception as e:
        errors.append(f"Error loading {p_s2.name}: {str(e)}")

    # 3. Complaint State Predictor v3
    p_state = resolve_path(MODELS_DIR, "complaint_predictor_v3.pkl")
    try:
        if p_state.exists():
            st_data = joblib.load(p_state)
            MODELS["state_model"] = st_data["model"]
            MODELS["state_classes"] = list(st_data["classes"])
            MODELS["label_encoders"] = st_data["label_encoders"]
            logger.info(f"Loaded Complaint Predictor: {p_state.name}")
        else:
            errors.append(f"Missing {p_state}")
    except Exception as e:
        errors.append(f"Error loading {p_state.name}: {str(e)}")

    # 4. District Risk Model & Scaler
    p_dist = resolve_path(MODELS_DIR, "district_model_v3.pkl")
    p_scaler = resolve_path(MODELS_DIR, "district_scaler_v3.pkl")
    try:
        if p_dist.exists():
            MODELS["district_model"] = joblib.load(p_dist)
            logger.info(f"Loaded District Model: {p_dist.name}")
        else:
            errors.append(f"Missing {p_dist}")
        if p_scaler.exists():
            MODELS["district_scaler"] = joblib.load(p_scaler)
            logger.info(f"Loaded District Scaler: {p_scaler.name}")
        else:
            errors.append(f"Missing {p_scaler}")
    except Exception as e:
        errors.append(f"Error loading district assets: {str(e)}")

    # 5. District Risk Scores CSV
    p_csv = resolve_path(DATA_DIR, "fixed_district_risk_scores_v3.csv")
    try:
        if p_csv.exists():
            df = pd.read_csv(p_csv)
            # Standardize column naming
            if "risk_score_v3" in df.columns:
                df["risk_score"] = df["risk_score_v3"].round(2)
            if "risk_tier_v3" in df.columns:
                df["risk_tier"] = df["risk_tier_v3"]
            MODELS["risk_df"] = df
            logger.info(f"Loaded District Risk Table: {p_csv.name} ({len(df)} records)")
        else:
            errors.append(f"Missing {p_csv}")
    except Exception as e:
        errors.append(f"Error loading risk CSV: {str(e)}")

    MODELS["errors"] = errors
    MODELS["loaded"] = len(errors) == 0
    if MODELS["loaded"]:
        logger.info("All ML models and datasets initialized successfully.")
    else:
        logger.warning(f"ML subsystem started in DEGRADED mode with {len(errors)} issues: {errors}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Cybercast ML Engine...")
    load_ml_assets()
    yield
    # Shutdown
    logger.info("Shutting down Cybercast ML Engine...")

# -----------------------------------------------------------------------------
# FASTAPI APPLICATION SETUP
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Cybercast Predictive Analytics API",
    description="Predicts cash withdrawal states, ATM zones, and district risks for cybercrime complaints (MHA / I4C PS 184)",
    version="3.0.0",
    lifespan=lifespan
)

# -----------------------------------------------------------------------------
# CORS CONFIGURATION (Allows Vercel, wildcards, and local development)
# -----------------------------------------------------------------------------
origins = [
    "https://184-two.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https://184-two.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# -----------------------------------------------------------------------------
class ComplaintRequest(BaseModel):
    complaint_id: Optional[str] = Field(None, example="NCRP-2024-MH-94812")
    fraud_type: str = Field(..., example="KYC_Fraud")
    amount_stolen_inr: float = Field(..., example=75000.0)
    victim_state: str = Field(..., example="Maharashtra")
    victim_district: Optional[str] = Field(None, example="Pune")
    victim_city_type: Optional[str] = Field("Metro", example="Metro")
    fraudster_phone_circle: Optional[str] = Field(None, example="Rajasthan")
    mule_account_bank: Optional[str] = Field(None, example="SBI")
    mule_account_state: Optional[str] = Field(None, example="Rajasthan")
    complaint_hour: Optional[int] = Field(None, ge=0, le=23, example=14)
    complaint_day_of_week: Optional[int] = Field(None, ge=0, le=6, example=2)
    complaint_timestamp: Optional[str] = Field(None, example="2024-07-15 14:30:00")

class StatePrediction(BaseModel):
    rank: int
    state: str
    probability: float

class ZonePrediction(BaseModel):
    predicted_zone: str
    confidence: float
    hierarchy_stage: str
    is_bank_counter: bool

class PredictionResponse(BaseModel):
    status: str
    complaint_id: str
    top_predicted_states: List[StatePrediction]
    zone_prediction: ZonePrediction
    estimated_time_window_hours: float
    time_urgency: str
    origin_district_risk: Optional[Dict[str, Any]]
    recommended_actions: List[str]
    processing_latency_ms: float

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def safe_encode(le, val):
    val_str = str(val) if val is not None else "Unknown"
    if hasattr(le, "classes_"):
        if val_str in le.classes_:
            return int(le.transform([val_str])[0])
        elif "Unknown" in le.classes_:
            return int(le.transform(["Unknown"])[0])
    return 0

def categorize_amount(amt: float) -> str:
    if amt < 10000:
        return "Small"
    elif amt < 100000:
        return "Medium"
    elif amt < 1000000:
        return "Large"
    return "VeryLarge"

def get_time_period_code(hour: int) -> int:
    if 6 <= hour < 10:
        return 0
    elif 10 <= hour < 17:
        return 1
    elif 17 <= hour < 22:
        return 2
    return 3

# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Cybercast Predictive Analytics API",
        "version": "3.0.0",
        "hackathon": "Smart India Hackathon 2024 (Problem Statement 184)",
        "models_status": "healthy" if MODELS["loaded"] else "degraded",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "districts": "GET /districts",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
async def health():
    """Railway healthcheck endpoint."""
    is_healthy = MODELS["loaded"]
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_200_OK  # Return 200 so container does not flap, but indicate status
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if is_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "models_loaded": {
                "zone_stage1": MODELS["zone_stage1"] is not None,
                "zone_stage2": MODELS["zone_stage2_model"] is not None,
                "state_predictor": MODELS["state_model"] is not None,
                "district_model": MODELS["district_model"] is not None,
                "district_risk_table": MODELS["risk_df"] is not None
            },
            "issues": MODELS["errors"]
        }
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_complaint(complaint: ComplaintRequest):
    """
    Predict cash withdrawal location (State & ATM Zone) and risk analysis for a complaint.
    """
    if not MODELS["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Models are not fully loaded: {MODELS['errors']}"
        )

    t0 = datetime.now()
    amt = float(complaint.amount_stolen_inr)
    amt_log = float(np.log1p(amt))
    amt_cat = categorize_amount(amt)
    
    # Infer hours if missing
    c_hour = complaint.complaint_hour
    c_day = complaint.complaint_day_of_week
    if c_hour is None or c_day is None:
        if complaint.complaint_timestamp:
            try:
                dt = pd.to_datetime(complaint.complaint_timestamp)
                c_hour = int(dt.hour) if c_hour is None else c_hour
                c_day = int(dt.dayofweek) if c_day is None else c_day
            except Exception:
                c_hour = 12 if c_hour is None else c_hour
                c_day = 2 if c_day is None else c_day
        else:
            c_hour = 12 if c_hour is None else c_hour
            c_day = 2 if c_day is None else c_day

    mule_state = complaint.mule_account_state or complaint.victim_state
    same_state = int(complaint.victim_state.strip().lower() == mule_state.strip().lower())
    is_urban = int(str(complaint.victim_city_type).strip() in ["Metro", "Tier2"])

    # 1. State Prediction
    state_model = MODELS["state_model"]
    state_classes = MODELS["state_classes"]
    le = MODELS["label_encoders"]

    state_feat = pd.DataFrame([{
        "fraud_type_encoded": safe_encode(le.get("fraud_type"), complaint.fraud_type),
        "victim_state_encoded": safe_encode(le.get("victim_state"), complaint.victim_state),
        "victim_city_type_encoded": safe_encode(le.get("victim_city_type"), complaint.victim_city_type),
        "fraudster_phone_circle_encoded": safe_encode(le.get("fraudster_phone_circle"), complaint.fraudster_phone_circle or mule_state),
        "mule_account_bank_encoded": safe_encode(le.get("mule_account_bank"), complaint.mule_account_bank or "SBI"),
        "mule_account_state_encoded": safe_encode(le.get("mule_account_state"), mule_state),
        "amount_category_encoded": safe_encode(le.get("amount_category"), amt_cat),
        "amount_stolen_log": amt_log,
        "complaint_hour": c_hour,
        "complaint_day_of_week": c_day,
        "same_state_withdrawal": same_state,
        "is_urban_victim": is_urban
    }])

    state_probs = state_model.predict_proba(state_feat)[0]
    top3_idx = np.argsort(state_probs)[::-1][:3]
    top_states = [
        StatePrediction(
            rank=i + 1,
            state=str(state_classes[idx]),
            probability=round(float(state_probs[idx]), 4)
        )
        for i, idx in enumerate(top3_idx)
    ]

    # 2. Hierarchical Zone Prediction
    s1 = MODELS["zone_stage1"]
    s2_model = MODELS["zone_stage2_model"]
    s2_classes = MODELS["zone_stage2_classes"]

    amt_vs_limit = amt / 200000.0
    is_above_limit = int(amt > 200000)
    is_above_double = int(amt > 400000)
    speed_map = {"OTP_Fraud": 1, "UPI_Fraud": 2, "KYC_Fraud": 3, "Loan_Fraud": 4, "Job_Fraud": 5, "Sextortion": 6, "Investment_Fraud": 7}
    fraud_speed = speed_map.get(complaint.fraud_type, 3)
    
    # Expected withdrawal timing
    w_hour = (c_hour + 2) % 24
    w_day = c_day
    time_p = get_time_period_code(w_hour)
    is_night = int(w_hour >= 22 or w_hour < 6)

    zone_feat = pd.DataFrame([{
        "amount_vs_atm_limit": amt_vs_limit,
        "is_above_atm_limit": is_above_limit,
        "is_above_double_limit": is_above_double,
        "fraud_speed_indicator": fraud_speed,
        "time_period_encoded": time_p,
        "same_state_withdrawal": same_state,
        "is_urban_victim": is_urban,
        "is_night": is_night,
        "amount_x_fraud_speed": amt_vs_limit * fraud_speed,
        "night_x_urban": is_night * is_urban,
        "interstate_x_amount": (1 - same_state) * amt_vs_limit,
        "high_amount_investment": int(amt > 200000 and complaint.fraud_type == "Investment_Fraud"),
        "fraud_type_encoded": safe_encode(le.get("fraud_type"), complaint.fraud_type),
        "amount_stolen_log": amt_log,
        "withdrawal_hour_of_day": w_hour,
        "withdrawal_day_of_week": w_day
    }])

    # Stage 1: ATM vs Bank Counter
    s1_counter_prob = float(s1.predict_proba(zone_feat)[0][1])
    if s1_counter_prob > 0.5:
        zone_info = ZonePrediction(
            predicted_zone="Bank_Branch_Counter",
            confidence=round(s1_counter_prob, 4),
            hierarchy_stage="Stage 1 (High Amount Over-the-Counter)",
            is_bank_counter=True
        )
    else:
        # Stage 2: ATM Subtype
        s2_probs = s2_model.predict_proba(zone_feat)[0]
        best_z_idx = int(np.argmax(s2_probs))
        zone_info = ZonePrediction(
            predicted_zone=str(s2_classes[best_z_idx]),
            confidence=round(float(s2_probs[best_z_idx]) * (1.0 - s1_counter_prob), 4),
            hierarchy_stage="Stage 2 (ATM Subtype Selection)",
            is_bank_counter=False
        )

    # 3. Estimated Time Window
    # Empirical rules: OTP/UPI fast (< 2h), Investment slower (24h+)
    time_est_map = {
        "OTP_Fraud": 1.2,
        "UPI_Fraud": 1.5,
        "KYC_Fraud": 3.0,
        "Job_Fraud": 5.0,
        "Loan_Fraud": 6.5,
        "Sextortion": 8.0,
        "Investment_Fraud": 36.0
    }
    est_hours = time_est_map.get(complaint.fraud_type, 4.0)
    if amt > 300000:
        est_hours += 8.0

    if est_hours <= 2.0:
        urgency = "CRITICAL (Cash-out expected in < 2 hours)"
    elif est_hours <= 6.0:
        urgency = "HIGH (Action window: 2 to 6 hours)"
    elif est_hours <= 18.0:
        urgency = "MODERATE (Interstate transfer in progress)"
    else:
        urgency = "DELAYED (Multi-day mule layering)"

    # 4. District Risk Lookup
    origin_risk = None
    if complaint.victim_district and MODELS["risk_df"] is not None:
        rdf = MODELS["risk_df"]
        v_dist_clean = complaint.victim_district.strip().lower()
        v_state_clean = complaint.victim_state.strip().lower()
        
        # Try exact match first
        matches = rdf[
            (rdf["State"].str.lower() == v_state_clean) &
            (rdf["District"].str.lower() == v_dist_clean)
        ]
        # Fallback to substring match if district has suffixes (e.g. Pune Commr / Pune Rural)
        if matches.empty:
            matches = rdf[
                (rdf["State"].str.lower() == v_state_clean) &
                (rdf["District"].str.lower().str.contains(v_dist_clean, regex=False, na=False))
            ]
        if not matches.empty:
            row = matches.iloc[0]
            origin_risk = {
                "state": row["State"],
                "district": row["District"],
                "risk_score": float(row["risk_score"]),
                "risk_tier": str(row["risk_tier"])
            }

    # 5. Tactical Recommendations
    top_target_state = top_states[0].state
    recommendations = []
    if zone_info.is_bank_counter:
        recommendations.append(f"Deploy branch interdiction: Alert banks in {top_target_state} for counter transactions exceeding ₹2,00,000.")
        recommendations.append(f"Request immediate lien/freeze on target mule account ({complaint.mule_account_bank or 'Partner Bank'}).")
    else:
        recommendations.append(f"Alert PCR patrol units near {zone_info.predicted_zone.replace('_', ' ')} clusters in {top_target_state}.")
        recommendations.append(f"Initiate CCTV sweep and monitoring during the {urgency.split('(')[0].strip()} window.")

    elapsed_ms = (datetime.now() - t0).total_seconds() * 1000.0

    return PredictionResponse(
        status="success",
        complaint_id=complaint.complaint_id or f"NCRP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        top_predicted_states=top_states,
        zone_prediction=zone_info,
        estimated_time_window_hours=round(est_hours, 1),
        time_urgency=urgency,
        origin_district_risk=origin_risk,
        recommended_actions=recommendations,
        processing_latency_ms=round(elapsed_ms, 2)
    )

@app.get("/districts", tags=["Heatmap Data"])
async def get_all_districts(state: Optional[str] = None):
    """
    Returns district cash-out risk scores for heatmap visualization on the frontend.
    """
    if MODELS["risk_df"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="District risk table not loaded."
        )
    df = MODELS["risk_df"]
    if state:
        df = df[df["State"].str.lower() == state.strip().lower()]
    
    records = df[["State", "District", "risk_score", "risk_tier"]].to_dict(orient="records")
    return {
        "count": len(records),
        "districts": records
    }

@app.get("/district/{state}/{district}", tags=["Heatmap Data"])
async def get_single_district(state: str, district: str):
    """Lookup risk score and tier for a single district."""
    if MODELS["risk_df"] is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data not loaded.")
    df = MODELS["risk_df"]
    match = df[
        (df["State"].str.lower() == state.strip().lower()) &
        (df["District"].str.lower() == district.strip().lower())
    ]
    if match.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="District not found.")
    row = match.iloc[0]
    return {
        "state": row["State"],
        "district": row["District"],
        "risk_score": float(row["risk_score"]),
        "risk_tier": str(row["risk_tier"])
    }

# -----------------------------------------------------------------------------
# MAIN RUNNER (For Railway or Local execution)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Railway passes the assigned port in the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Binding server to port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
