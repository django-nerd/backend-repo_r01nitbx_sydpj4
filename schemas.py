"""
Database Schemas for CivicPulse

Each Pydantic model represents a MongoDB collection (collection name is the lowercase of class name).
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

# Core content captured from external sources (twitter, reddit, news, forums)
class Post(BaseModel):
    source: str = Field(..., description="Source platform e.g., twitter, reddit, news, forum")
    external_id: Optional[str] = Field(None, description="ID from the source platform")
    author: Optional[str] = None
    text: str
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    language: Optional[str] = "en"
    created_at_source: Optional[datetime] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    sentiment_label: Optional[str] = Field(None, description="negative|neutral|positive")
    topics: List[str] = []
    meta: Dict[str, Any] = {}

class Complaint(BaseModel):
    description: str
    city: Optional[str] = None
    category: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    image_url: Optional[str] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    sentiment_label: Optional[str] = None
    status: str = Field("new", description="new|reviewed|resolved")
    submitted_by: Optional[EmailStr] = None

class TopicCluster(BaseModel):
    title: str
    keywords: List[str] = []
    sample_posts: List[str] = []
    city: Optional[str] = None
    sentiment_breakdown: Dict[str, int] = Field(default_factory=lambda: {"positive":0, "neutral":0, "negative":0})
    volume: int = 0
    time_window: str = Field("24h", description="Aggregation window e.g., 24h, 7d")

class User(BaseModel):
    name: str
    email: EmailStr
    role: str = Field("citizen", description="citizen|analyst|admin")
    is_active: bool = True

class IngestionMetric(BaseModel):
    source: str
    count: int
    timeframe: str = "1h"

# Response models
class SentimentPoint(BaseModel):
    lat: float
    lon: float
    score: float
    label: str
    city: Optional[str] = None
    text: Optional[str] = None

class AnalyzeRequest(BaseModel):
    text: str
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

class ComplaintRequest(BaseModel):
    description: str
    city: Optional[str] = None
    category: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    image_url: Optional[str] = None
    submitted_by: Optional[EmailStr] = None
