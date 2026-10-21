from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal, Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import get_db
from .schemas import (
    GroupResponse,
    HealthResponse,
    HouseholdDetail,
    MetaResponse,
    OverviewResponse,
    ValidationReport,
)
from .seed import init_database
from .services import DIMENSIONS, get_group_metrics, get_household, get_overview, get_validation_report


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    yield


app = FastAPI(
    title="CFPS Data Support Platform (Demo)",
    description=(
        "A small data-support API for the CFPS rural digital-service analysis project. "
        "The database is seeded with synthetic demo rows; CFPS microdata is not distributed."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

Dimension = Literal["internet", "pc_internet", "gender", "age_group", "education", "income_quartile"]
DbSession = Annotated[Session, Depends(get_db)]


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "CFPS Data Support Platform (Demo)",
        "docs": "/docs",
        "health": "/health",
        "data_mode": "demo",
        "notice": "Synthetic demo rows only; CFPS microdata is not included.",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(db: DbSession):
    db.execute(text("SELECT 1"))
    return HealthResponse(status="ok", database="sqlite", data_mode="demo")


@app.get("/api/v1/meta", response_model=MetaResponse, tags=["meta"])
def meta():
    return MetaResponse(
        name="CFPS Data Support Platform",
        version="0.1.0",
        data_mode="demo",
        data_notice="Synthetic demo rows only; CFPS microdata is not distributed.",
        docs_url="/docs",
    )


@app.get("/api/v1/metrics/overview", response_model=OverviewResponse, tags=["metrics"])
def overview(db: DbSession):
    return get_overview(db)


@app.get("/api/v1/metrics/groups", response_model=GroupResponse, tags=["metrics"])
def groups(
    db: DbSession,
    dimension: Dimension = Query(
        "internet",
        description="Grouping dimension. One of: internet, pc_internet, gender, age_group, education, income_quartile.",
    ),
):
    if dimension not in DIMENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported dimension: {dimension}")
    return get_group_metrics(db, dimension)


@app.get("/api/v1/households/{fid22}", response_model=HouseholdDetail, tags=["households"])
def household(fid22: int, db: DbSession):
    item = get_household(db, fid22)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Household {fid22} not found")
    return item


@app.get("/api/v1/validation", response_model=ValidationReport, tags=["validation"])
def validation(db: DbSession):
    return get_validation_report(db)
