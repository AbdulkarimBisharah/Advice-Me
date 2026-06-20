"""
Advisor Co-Pilot — Intelligence API
Run: uvicorn backend_app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import os
import pathlib
import joblib
import logging
import numpy as np
import pandas as pd
import shap
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional
import anthropic

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("copilot-api")

# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------
BASE = pathlib.Path(__file__).parent

# Churn — LightGBM model from Colab + matching preprocessors
_churn_model    = None   # LightGBM classifier
_churn_scaler   = None   # StandardScaler
_churn_encoders = None   # dict[col] -> LabelEncoder
_churn_columns  = None   # ordered feature list
_churn_explainer = None  # shap.TreeExplainer

# Complaint classifier
_complaint_vec = None
_complaint_clf = None


def load_churn():
    global _churn_model, _churn_scaler, _churn_encoders, _churn_columns, _churn_explainer
    if _churn_model is None:
        _churn_model    = joblib.load(BASE / "churn_model.joblib")
        _churn_scaler   = joblib.load(BASE / "churn_preprocessor.joblib")
        _churn_encoders = joblib.load(BASE / "churn_label_encoders.joblib")
        _churn_columns  = joblib.load(BASE / "churn_feature_columns.joblib")
        _churn_explainer = shap.TreeExplainer(_churn_model)
        log.info("Churn model (LightGBM) loaded — %d features", len(_churn_columns))


def load_complaint():
    global _complaint_vec, _complaint_clf
    if _complaint_vec is None:
        _complaint_vec = joblib.load(BASE / "complaint_vectorizer.pkl")
        _complaint_clf = joblib.load(BASE / "complaint_model.pkl")
        log.info("Complaint classifier loaded (%d classes)", len(_complaint_clf.classes_))


# ---------------------------------------------------------------------------
# Startup: preload models so the first request isn't slow
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    for loader, name in [(load_churn, "churn"), (load_complaint, "complaint")]:
        try:
            loader()
        except FileNotFoundError:
            log.warning("%s model pkl not found — run training script first", name)
        except Exception as exc:
            log.error("Failed to load %s model: %s", name, exc)
    yield


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Advisor Co-Pilot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Churn column defaults (medians/modes from training distribution)
# ---------------------------------------------------------------------------
CHURN_DEFAULTS: Dict[str, Any] = {
    "region_name": "Auckland",
    "age": 45,
    "age_band": "45-54",
    "marital_status": "Married",
    "customer_tenure_months": 60,
    "multi_policy_flag": 0,
    "num_policies": 1,
    "policy_type": "Auto",
    "renewal_month": 6,
    "current_premium": 900.0,
    "premium_last_year": 870.0,
    "premium_change_pct": 0.03,
    "num_price_increases_last_3y": 1,
    "coverage_amount": 30000.0,
    "premium_to_coverage_ratio": 0.03,
    "payment_frequency": "Monthly",
    "autopay_enabled": 1,
    "late_payment_count_12m": 0,
    "missed_payment_flag": 0,
    "payment_method_change_flag": 0,
    "num_claims_12m": 0,
    "num_approved_claims_12m": 0,
    "num_rejected_claims_12m": 0,
    "num_pending_claims_12m": 0,
    "avg_claim_amount": 2000.0,
    "total_claim_amount_12m": 0.0,
    "total_payout_amount_12m": 0.0,
    "payout_ratio_12m": 0.75,
    "avg_settlement_time_days": 10,
    "days_since_last_claim": 365,
    "num_contacts_12m": 2,
    "complaint_flag": 0,
    "complaint_resolution_days": 0,
    "quote_requested_flag": 0,
    "coverage_downgrade_flag": 0,
}


def _risk_band(prob: float) -> str:
    # Thresholds match the Colab notebook (risk_band function in cell 2)
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def _age_band(age: int) -> str:
    # FIX: upper bound is the top of each band, not the bottom of next
    bands = [(24, "18-24"), (34, "25-34"), (44, "35-44"), (54, "45-54"),
             (64, "55-64"), (74, "65-74")]
    for upper, band in bands:
        if age <= upper:
            return band
    return "75+"


def _heuristic_churn(p: "RiskInput") -> float:
    score = 0.20
    if p.complaint_flag:
        score += 0.30
    if p.late_payment_count_12m and p.late_payment_count_12m >= 2:
        score += 0.15
    if p.quote_requested_flag:
        score += 0.20
    if p.coverage_downgrade_flag:
        score += 0.15
    if p.num_contacts_12m is not None and p.num_contacts_12m == 0:
        score += 0.10
    if p.days_since_contact is not None and p.days_since_contact > 30:
        score += 0.10
    return min(score, 0.97)


def _build_churn_row(payload: "RiskInput") -> pd.DataFrame:
    """Build and preprocess one row exactly as the Colab notebook does:
    1. Fill defaults for missing fields
    2. Apply LabelEncoder per categorical column
    3. Apply StandardScaler
    """
    row = dict(CHURN_DEFAULTS)

    # Map API fields onto training columns
    if payload.age is not None:
        row["age"] = payload.age
        row["age_band"] = _age_band(payload.age)
    if payload.monthly_income is not None:
        row["current_premium"] = payload.monthly_income * 0.12
        row["premium_last_year"] = row["current_premium"]
    if payload.complaint_flag is not None:
        row["complaint_flag"] = payload.complaint_flag
    if payload.num_contacts_12m is not None:
        row["num_contacts_12m"] = payload.num_contacts_12m
    if payload.late_payment_count_12m is not None:
        row["late_payment_count_12m"] = payload.late_payment_count_12m
        row["missed_payment_flag"] = int(payload.late_payment_count_12m >= 4)
    if payload.num_claims_12m is not None:
        row["num_claims_12m"] = payload.num_claims_12m
    if payload.quote_requested_flag is not None:
        row["quote_requested_flag"] = payload.quote_requested_flag
    if payload.coverage_downgrade_flag is not None:
        row["coverage_downgrade_flag"] = payload.coverage_downgrade_flag
    if payload.customer_tenure_months is not None:
        row["customer_tenure_months"] = payload.customer_tenure_months
    if payload.days_since_contact is not None:
        row["days_since_last_claim"] = payload.days_since_contact
    if payload.extra:
        for k, v in payload.extra.items():
            if k in _churn_columns:
                row[k] = v

    df = pd.DataFrame([row])

    # Step 1: LabelEncode categoricals (mirror Colab cell 2)
    for col, le in _churn_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # Ensure correct column order; fill any missing with 0
    for col in _churn_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[_churn_columns]

    # Step 2: StandardScaler (mirror Colab cell 2)
    df_scaled = pd.DataFrame(
        _churn_scaler.transform(df),
        columns=_churn_columns,
    )
    return df_scaled


# ---------------------------------------------------------------------------
# /predict-risk  (single client)
# ---------------------------------------------------------------------------
class RiskInput(BaseModel):
    age: Optional[int] = None
    monthly_income: Optional[float] = None
    complaint_flag: Optional[int] = Field(None, description="1 if complaint raised")
    num_contacts_12m: Optional[int] = None
    late_payment_count_12m: Optional[int] = None
    num_claims_12m: Optional[int] = None
    quote_requested_flag: Optional[int] = None
    coverage_downgrade_flag: Optional[int] = None
    customer_tenure_months: Optional[int] = None
    days_since_contact: Optional[int] = Field(None, ge=0, description="Days since last advisor contact")
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("age")
    @classmethod
    def age_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("age must be positive")
        return v

    @field_validator("monthly_income")
    @classmethod
    def income_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError("monthly_income must be non-negative")
        return v


class RiskOutput(BaseModel):
    churn_probability: float
    risk_band: str
    top_risk_drivers: List[str] = Field(default_factory=list, description="Top 3 SHAP feature drivers")


def _shap_drivers(df_scaled: pd.DataFrame) -> List[str]:
    try:
        shap_vals = _churn_explainer.shap_values(df_scaled)
        # LightGBM binary returns list[ndarray] or single ndarray depending on version
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        return (
            pd.Series(np.abs(sv[0]), index=_churn_columns)
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
    except Exception as exc:
        log.warning("SHAP computation failed: %s", exc)
        return []


@app.post("/predict-risk", response_model=RiskOutput, tags=["ML"])
def predict_risk(payload: RiskInput):
    try:
        load_churn()
    except Exception:
        prob = _heuristic_churn(payload)
        return RiskOutput(churn_probability=round(prob, 4), risk_band=_risk_band(prob))

    try:
        df = _build_churn_row(payload)
        prob = float(_churn_model.predict_proba(df)[0, 1])
        drivers = _shap_drivers(df)
        return RiskOutput(churn_probability=round(prob, 4), risk_band=_risk_band(prob), top_risk_drivers=drivers)
    except Exception as exc:
        log.error("predict-risk model error: %s", exc)
        prob = _heuristic_churn(payload)
        return RiskOutput(churn_probability=round(prob, 4), risk_band=_risk_band(prob))


# ---------------------------------------------------------------------------
# /batch-risk  (score multiple clients in one call — needed by Jarvis Briefing
#              and Constellation View so the frontend avoids N serial requests)
# ---------------------------------------------------------------------------
class BatchRiskItem(RiskInput):
    client_id: str = Field(..., description="Client identifier (pass Supabase row id)")


class BatchRiskOutput(BaseModel):
    client_id: str
    churn_probability: float
    risk_band: str
    top_risk_drivers: List[str] = Field(default_factory=list)


@app.post("/batch-risk", response_model=List[BatchRiskOutput], tags=["ML"])
def batch_risk(clients: List[BatchRiskItem]):
    if not clients:
        return []
    if len(clients) > 200:
        raise HTTPException(status_code=400, detail="Max 200 clients per batch")

    churn_model_ok = True
    try:
        load_churn()
    except Exception:
        churn_model_ok = False

    results = []
    for item in clients:
        try:
            if churn_model_ok:
                df = _build_churn_row(item)
                prob = float(_churn_model.predict_proba(df)[0, 1])
                drivers = _shap_drivers(df)
            else:
                prob = _heuristic_churn(item)
                drivers = []
        except Exception as exc:
            log.error("batch-risk error for %s: %s", item.client_id, exc)
            prob = _heuristic_churn(item)
            drivers = []
        results.append(BatchRiskOutput(
            client_id=item.client_id,
            churn_probability=round(prob, 4),
            risk_band=_risk_band(prob),
            top_risk_drivers=drivers,
        ))
    return results


# ---------------------------------------------------------------------------
# /classify
# ---------------------------------------------------------------------------
class ClassifyInput(BaseModel):
    text: str = Field(..., min_length=1, description="Complaint or concern narrative")


class ClassifyOutput(BaseModel):
    reason: str
    confidence: float


@app.post("/classify", response_model=ClassifyOutput, tags=["ML"])
def classify(payload: ClassifyInput):
    try:
        load_complaint()
    except Exception:
        return ClassifyOutput(reason="General complaint", confidence=0.5)

    try:
        vec = _complaint_vec.transform([payload.text])
        proba = _complaint_clf.predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        label = _complaint_clf.classes_[idx]
        return ClassifyOutput(reason=label, confidence=round(float(proba[idx]), 4))
    except Exception as exc:
        log.error("classify error: %s", exc)
        return ClassifyOutput(reason="General complaint", confidence=0.5)


# ---------------------------------------------------------------------------
# /copilot
# ---------------------------------------------------------------------------
class CopilotInput(BaseModel):
    client_name: str = Field(..., min_length=1)
    risk_pct: float = Field(..., description="Churn probability 0–1")
    issue: Optional[str] = None
    note: Optional[str] = None
    days_since_contact: Optional[int] = Field(None, ge=0)
    advisor_name: Optional[str] = "the advisor"

    @field_validator("risk_pct")
    @classmethod
    def risk_in_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("risk_pct must be between 0 and 1")
        return v


class CopilotOutput(BaseModel):
    briefing: str
    follow_up_draft: str


def _build_copilot_prompt(payload: CopilotInput) -> str:
    silence_str = (
        f", last contacted {payload.days_since_contact} days ago"
        if payload.days_since_contact is not None
        else ""
    )
    issue_str = f"\n- Raised issue: {payload.issue}" if payload.issue else ""
    note_str = f"\n- Advisor note: {payload.note}" if payload.note else ""

    return f"""You are an AI co-pilot for a financial advisor named {payload.advisor_name}.

Client summary:
- Name: {payload.client_name}
- Churn risk: {round(payload.risk_pct * 100, 1)}%{silence_str}{issue_str}{note_str}

Provide exactly two things, clearly labelled:

BRIEFING: A single sentence (max 20 words) for the morning dashboard — plain language, urgent but calm. Include the client name, risk level, and key reason.

FOLLOW_UP_DRAFT: A short, warm, professional email (3–5 sentences) written in the advisor's voice. Do NOT mention churn risk or internal scores. Sound human, not automated.

Reply with BRIEFING: on one line, then FOLLOW_UP_DRAFT: followed by the email body. Nothing else."""


def _parse_copilot_response(raw: str, payload: CopilotInput):
    briefing = ""
    draft_lines: List[str] = []
    current = None

    for line in raw.strip().splitlines():
        upper = line.upper()
        if upper.startswith("BRIEFING:"):
            current = "briefing"
            briefing = line[len("BRIEFING:"):].strip()
        elif upper.startswith("FOLLOW_UP_DRAFT:"):
            current = "draft"
            first = line[len("FOLLOW_UP_DRAFT:"):].strip()
            if first:
                draft_lines.append(first)
        else:
            if current == "draft":
                draft_lines.append(line)
            elif current == "briefing" and not briefing:
                briefing = line.strip()

    draft = "\n".join(draft_lines).strip()

    if not briefing:
        briefing = _copilot_fallback(payload).briefing
    if not draft:
        draft = _copilot_fallback(payload).follow_up_draft
    return briefing, draft


def _copilot_fallback(payload: CopilotInput) -> CopilotOutput:
    silence_str = (
        f", silent for {payload.days_since_contact} days"
        if payload.days_since_contact
        else ""
    )
    briefing = (
        f"{payload.client_name} — {round(payload.risk_pct * 100, 1)}% lapse risk"
        f"{silence_str}"
        + (f", raised: {payload.issue}" if payload.issue else "")
        + ". Reach out today."
    )
    draft = (
        f"Hi {payload.client_name},\n\n"
        "I wanted to personally reach out and check in on how you're doing. "
        "It's been a while since we last caught up and I'd love to hear how things are going for you. "
        "Please feel free to give me a call or reply to this message — I'm here to help.\n\n"
        f"Warm regards,\n{payload.advisor_name}"
    )
    return CopilotOutput(briefing=briefing, follow_up_draft=draft)


@app.post("/copilot", response_model=CopilotOutput, tags=["LLM"])
def copilot(payload: CopilotInput):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("/copilot called but ANTHROPIC_API_KEY is not set — using fallback")
        return _copilot_fallback(payload)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": _build_copilot_prompt(payload)}],
        )
        raw = "".join(
            block.text for block in message.content if hasattr(block, "text")
        )
        briefing, draft = _parse_copilot_response(raw, payload)
        return CopilotOutput(briefing=briefing, follow_up_draft=draft)
    except Exception as exc:
        log.error("/copilot LLM error: %s", exc)
        return _copilot_fallback(payload)


# ---------------------------------------------------------------------------
# /parse-task
# ---------------------------------------------------------------------------
class ParseTaskInput(BaseModel):
    text: str = Field(..., min_length=1, description="Free-text from voice or chat")


class ParseTaskOutput(BaseModel):
    category: str = Field(..., description="deadline|followup|client_meeting|team_meeting")
    title: str
    due_date: Optional[str] = Field(None, description="ISO date YYYY-MM-DD or null")


_TASK_CATEGORIES = ["deadline", "followup", "client_meeting", "team_meeting"]

_TASK_KEYWORDS: Dict[str, List[str]] = {
    "client_meeting": ["meet with client", "meeting with", "client call", "call with", "lunch with", "sit down with"],
    "team_meeting": ["team meeting", "team call", "standup", "stand-up", "sync with team", "weekly meeting"],
    "deadline": ["submit", "deadline", "due", "file", "report", "complete by", "finish by", "send by"],
    "followup": ["follow up", "follow-up", "check in", "reach out", "contact", "email", "call"],
}

_PARSE_TASK_PROMPT_PREFIX = (
    "Extract the task from this text: "
)
_PARSE_TASK_PROMPT_SUFFIX = (
    "\n\nReply with exactly three lines:\n"
    "CATEGORY: one of deadline|followup|client_meeting|team_meeting\n"
    "TITLE: short task title (max 8 words)\n"
    "DUE_DATE: ISO date like 2026-06-25 or null\n"
    "Nothing else."
)


def _build_parse_task_prompt(text: str) -> str:
    # Use string concatenation instead of .format() so curly braces in user
    # text (e.g. "{policy}") don't raise KeyError
    return _PARSE_TASK_PROMPT_PREFIX + '"' + text + '"' + _PARSE_TASK_PROMPT_SUFFIX


def _parse_task_response(raw: str, original: str) -> ParseTaskOutput:
    category = "followup"
    title = original[:60]
    due_date = None
    for line in raw.strip().splitlines():
        upper = line.upper()
        if upper.startswith("CATEGORY:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in _TASK_CATEGORIES:
                category = val
        elif upper.startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()
        elif upper.startswith("DUE_DATE:"):
            val = line.split(":", 1)[1].strip()
            due_date = None if val.lower() == "null" else val
    return ParseTaskOutput(category=category, title=title, due_date=due_date)


def _heuristic_task(text: str) -> ParseTaskOutput:
    lower = text.lower()
    category = "followup"
    for cat, keywords in _TASK_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            category = cat
            break
    return ParseTaskOutput(category=category, title=text.strip()[:60], due_date=None)


@app.post("/parse-task", response_model=ParseTaskOutput, tags=["LLM"])
def parse_task(payload: ParseTaskInput):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": _build_parse_task_prompt(payload.text)}],
            )
            raw = "".join(b.text for b in message.content if hasattr(b, "text"))
            return _parse_task_response(raw, payload.text)
        except Exception as exc:
            log.error("/parse-task LLM error: %s", exc)

    return _heuristic_task(payload.text)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def health():
    return {
        "status": "ok",
        "churn_model": "ready (LightGBM + SHAP)" if _churn_model is not None else "not loaded — run generate_preprocessor.py and copy churn_model.joblib",
        "complaint_model": "ready" if _complaint_clf is not None else "not loaded — run 02_complaint_classifier.py",
        "anthropic_key": "set" if os.environ.get("ANTHROPIC_API_KEY") else "missing",
    }
