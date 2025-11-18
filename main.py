import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import (
    AnalyzeRequest,
    ComplaintRequest,
    Complaint,
    Post,
    TopicCluster,
    SentimentPoint,
)

app = FastAPI(title="CivicPulse API", description="Mapping Public Sentiment in Real Time")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "CivicPulse Backend Running"}

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = getattr(db, 'name', None) or ("✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set")
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Env echo
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

# -----------------------------
# Minimal NLP placeholder utils
# -----------------------------

class SentimentResult(BaseModel):
    score: float
    label: str


def simple_sentiment(text: str) -> SentimentResult:
    """Lightweight heuristic as a placeholder.
    Replace with transformer model in production.
    """
    text_l = text.lower()
    pos_words = ["good", "great", "love", "happy", "safe", "clean"]
    neg_words = ["bad", "hate", "angry", "unsafe", "dirty", "traffic", "crime", "water"]
    score = 0
    for w in pos_words:
        if w in text_l:
            score += 1
    for w in neg_words:
        if w in text_l:
            score -= 1
    score = max(-3, min(3, score)) / 3.0
    label = "positive" if score > 0.2 else ("negative" if score < -0.2 else "neutral")
    return SentimentResult(score=score, label=label)


# -----------------------------
# API Endpoints (spec compliant)
# -----------------------------

@app.post("/api/sentiment/analyze")
def analyze_text(payload: AnalyzeRequest) -> Dict[str, Any]:
    """Analyze a piece of text, optionally geotagged, and persist a Post document."""
    s = simple_sentiment(payload.text)
    post = Post(
        source="user",
        external_id=None,
        author=None,
        text=payload.text,
        city=payload.city,
        lat=payload.lat,
        lon=payload.lon,
        created_at_source=datetime.now(timezone.utc),
        sentiment_score=s.score,
        sentiment_label=s.label,
        topics=[],
    )
    try:
        inserted_id = create_document("post", post)
    except Exception:
        inserted_id = None
    return {
        "sentiment": {"score": s.score, "label": s.label},
        "saved": inserted_id is not None,
        "id": inserted_id,
    }

@app.get("/api/sentiment/city")
def sentiment_by_city(name: str, hours: int = 24) -> Dict[str, Any]:
    """Return recent sentiment points and simple aggregates for a city."""
    try:
        # Get latest posts/complaints with location
        posts = get_documents("post", {"city": name})
        complaints = get_documents("complaint", {"city": name})
    except Exception:
        posts, complaints = [], []

    points: List[Dict[str, Any]] = []
    for p in posts + complaints:
        if p.get("lat") is not None and p.get("lon") is not None and p.get("sentiment_score") is not None:
            points.append({
                "lat": p["lat"],
                "lon": p["lon"],
                "score": p["sentiment_score"],
                "label": p.get("sentiment_label", "neutral"),
                "city": p.get("city", name),
                "text": p.get("text") or p.get("description")
            })

    # Simple aggregate
    avg = 0.0
    if points:
        avg = sum(p["score"] for p in points) / len(points)
    agg_label = "positive" if avg > 0.2 else ("negative" if avg < -0.2 else "neutral")

    return {
        "city": name,
        "points": points[:1000],
        "summary": {"average": avg, "label": agg_label, "count": len(points)},
    }

@app.get("/api/sentiment/trending")
def trending_issues(city: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
    """Return basic mock clusters based on keyword heuristics.
    Replace with BERTopic/LDA pipeline later.
    """
    try:
        posts = get_documents("post", ({"city": city} if city else {}))
    except Exception:
        posts = []

    buckets = {
        "Traffic Congestion": ["traffic", "congestion", "roads", "jam"],
        "Water Problems": ["water", "supply", "leak", "sewage"],
        "Public Safety": ["crime", "police", "unsafe", "theft"],
        "Public Transport": ["bus", "metro", "train", "delay"],
    }

    clusters = []
    for title, kws in buckets.items():
        matched = [p for p in posts if any(k in (p.get("text", "").lower()) for k in kws)]
        if not matched:
            continue
        pos = sum(1 for p in matched if (p.get("sentiment_label") == "positive"))
        neg = sum(1 for p in matched if (p.get("sentiment_label") == "negative"))
        neu = sum(1 for p in matched if (p.get("sentiment_label") not in ("positive", "negative")))
        clusters.append({
            "title": title,
            "keywords": kws,
            "city": city,
            "sentiment_breakdown": {"positive": pos, "neutral": neu, "negative": neg},
            "volume": len(matched),
            "sample_posts": [p.get("text") for p in matched[:5]],
        })

    return {"city": city, "clusters": clusters}

@app.post("/api/complaints/new")
def new_complaint(payload: ComplaintRequest) -> Dict[str, Any]:
    """Accept a citizen complaint, run sentiment, store, and return the record."""
    s = simple_sentiment(payload.description)
    complaint = Complaint(
        description=payload.description,
        city=payload.city,
        category=payload.category,
        lat=payload.lat,
        lon=payload.lon,
        image_url=payload.image_url,
        submitted_by=payload.submitted_by,
        sentiment_score=s.score,
        sentiment_label=s.label,
        status="new",
    )
    try:
        inserted_id = create_document("complaint", complaint)
    except Exception:
        inserted_id = None

    return {
        "sentiment": {"score": s.score, "label": s.label},
        "saved": inserted_id is not None,
        "id": inserted_id,
    }

@app.get("/api/complaints/city")
def complaints_by_city(city: str, limit: int = 100) -> Dict[str, Any]:
    try:
        items = get_documents("complaint", {"city": city}, limit=limit)
    except Exception:
        items = []
    return {"city": city, "items": items}

@app.get("/api/analytics/overview")
def analytics_overview(city: Optional[str] = None) -> Dict[str, Any]:
    """Return minimal analytics across sources as a starting point."""
    try:
        post_count = len(get_documents("post", ({"city": city} if city else {})))
    except Exception:
        post_count = 0
    try:
        complaint_count = len(get_documents("complaint", ({"city": city} if city else {})))
    except Exception:
        complaint_count = 0

    # naive estimates for demo
    total = post_count + complaint_count
    pos_rate = 0.0
    neg_rate = 0.0
    if total:
        # derive from aggregates
        try:
            sample = get_documents("complaint", ({"city": city} if city else {}), limit=200) + \
                     get_documents("post", ({"city": city} if city else {}), limit=200)
        except Exception:
            sample = []
        if sample:
            pos = sum(1 for s in sample if s.get("sentiment_label") == "positive")
            neg = sum(1 for s in sample if s.get("sentiment_label") == "negative")
            pos_rate = pos / len(sample)
            neg_rate = neg / len(sample)

    return {
        "city": city,
        "sources": {
            "posts": post_count,
            "complaints": complaint_count,
        },
        "rates": {
            "positive": pos_rate,
            "negative": neg_rate,
        },
        "generated_at": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
