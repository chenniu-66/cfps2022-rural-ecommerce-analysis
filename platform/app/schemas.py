from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    database: str
    data_mode: str = "demo"


class MetaResponse(BaseModel):
    name: str
    version: str
    data_mode: str
    data_notice: str
    docs_url: str


class OverviewResponse(BaseModel):
    households: int = Field(ge=0)
    adults: int = Field(ge=0)
    matched_pairs: int = Field(ge=0)
    adopted_households: int = Field(ge=0)
    adoption_rate: float = Field(ge=0, le=1)


class AdultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    pid: int
    age: float | None = None
    male: int
    edu_year: float | None = None
    internet: int
    pc_internet: int


class HouseholdOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fid22: int
    resp1pid: int
    familysize: int | None = None
    ln_income_per: float | None = None
    service_adopt: int
    land_asset_log: float | None = None


class HouseholdDetail(BaseModel):
    household: HouseholdOut
    adult: AdultOut


class GroupMetric(BaseModel):
    dimension: str
    group: str
    n: int = Field(ge=0)
    adopted: int = Field(ge=0)
    adoption_rate: float = Field(ge=0, le=1)


class GroupResponse(BaseModel):
    dimension: str
    groups: list[GroupMetric]


class ValidationCheck(BaseModel):
    name: str
    status: str
    value: int | float
    expected: str


class ValidationReport(BaseModel):
    status: str
    generated_at: datetime
    checks: list[ValidationCheck]
