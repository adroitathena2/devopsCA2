"""Clinical Copilot — lightweight service wrapper for DevOps tasks.

Full ML pipeline (GLiNER + Granite embeddings + DrugBank) lives in
clinical_copilot/ and needs GPU + 2GB cache. This module exposes the same
contract (/analyze) with a rule-based fallback so CI / Docker / K8s /
Prometheus demos run in seconds without heavy deps.
"""
import os
import re
import time
from typing import List, Optional

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
START_TIME = time.time()

app = FastAPI(
    title="Clinical Copilot Service",
    description="Drug interaction screening API (lightweight DevOps wrapper)",
    version=APP_VERSION,
)

# ---- Prometheus metrics ----
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds", "Request latency seconds", ["endpoint"]
)
INTERACTIONS_TOTAL = Counter(
    "interactions_detected_total", "Total drug-drug interactions flagged"
)
ERRORS_TOTAL = Counter("app_errors_total", "Total application errors")
UPTIME_GAUGE = Gauge("app_uptime_seconds", "Application uptime in seconds")
APP_INFO = Gauge("app_info", "Application version info", ["version", "service"])


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Prescription text")


class Interaction(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    description: str
    detection_confidence: float


class AnalyzeResponse(BaseModel):
    identified_medicines: List[str]
    drug_interactions: List[Interaction]
    total_medicines: int
    total_interactions: int
    has_safety_concerns: bool
    version: str


# Minimal demo formulary (subset of DrugBank names used in README examples)
KNOWN_DRUGS = [
    "aspirin", "clopidogrel", "atorvastatin", "metformin",
    "glimepiride", "dolo", "paracetamol", "ibuprofen",
    "warfarin", "amlodipine", "losartan",
]

# Demo pairwise knowledge base: frozenset -> (severity, description)
INTERACTION_KB = {
    frozenset(["aspirin", "clopidogrel"]): (
        "Severe",
        "Increased risk of bleeding when antiplatelets are combined.",
    ),
    frozenset(["aspirin", "warfarin"]): (
        "Severe",
        "Markedly increased bleeding risk (anticoagulant + antiplatelet).",
    ),
    frozenset(["aspirin", "ibuprofen"]): (
        "Moderate",
        "Ibuprofen may blunt aspirin cardioprotection + GI risk.",
    ),
    frozenset(["metformin", "glimepiride"]): (
        "Moderate",
        "Additive hypoglycemic effect; monitor blood glucose.",
    ),
    frozenset(["amlodipine", "losartan"]): (
        "Mild",
        "Additive hypotension; usually intentional combination.",
    ),
    frozenset(["atorvastatin", "clopidogrel"]): (
        "Mild",
        "Possible reduced clopidogrel activation; monitor.",
    ),
}


def extract_medicines(text: str) -> List[str]:
    lowered = text.lower()
    found = []
    for drug in KNOWN_DRUGS:
        if re.search(r"\b" + re.escape(drug) + r"\b", lowered):
            # return canonical display name
            found.append(drug.capitalize() if drug != "dolo" else "Dolo")
    # de-dup preserving order
    return list(dict.fromkeys(found))


def detect_interactions(meds: List[str]) -> List[dict]:
    norm = [m.lower() for m in meds]
    out = []
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            key = frozenset([norm[i], norm[j]])
            if key in INTERACTION_KB:
                sev, desc = INTERACTION_KB[key]
                out.append(
                    {
                        "drug_a": meds[i],
                        "drug_b": meds[j],
                        "severity": sev,
                        "description": desc,
                        "detection_confidence": 0.9 if sev == "Severe" else 0.8,
                    }
                )
    return out


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception:
        ERRORS_TOTAL.inc()
        raise
    latency = time.time() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, status=status
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    UPTIME_GAUGE.set(time.time() - START_TIME)
    return response


@app.get("/")
def root():
    APP_INFO.labels(version=APP_VERSION, service="clinical-copilot").set(1)
    return {
        "service": "clinical-copilot",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    UPTIME_GAUGE.set(time.time() - START_TIME)
    return {
        "status": "up",
        "version": APP_VERSION,
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }


@app.get("/ready")
def ready():
    return {"ready": True, "version": APP_VERSION}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    meds = extract_medicines(req.text)
    interactions = detect_interactions(meds)
    INTERACTIONS_TOTAL.inc(len(interactions))
    return AnalyzeResponse(
        identified_medicines=meds,
        drug_interactions=[Interaction(**i) for i in interactions],
        total_medicines=len(meds),
        total_interactions=len(interactions),
        has_safety_concerns=any(
            i["severity"] == "Severe" for i in interactions
        ),
        version=APP_VERSION,
    )


@app.get("/metrics")
def metrics():
    UPTIME_GAUGE.set(time.time() - START_TIME)
    APP_INFO.labels(version=APP_VERSION, service="clinical-copilot").set(1)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
