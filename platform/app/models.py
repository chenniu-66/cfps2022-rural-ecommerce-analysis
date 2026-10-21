from __future__ import annotations

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Adult(Base):
    __tablename__ = "adults"
    __table_args__ = (
        CheckConstraint("age IS NULL OR (age >= 0 AND age <= 120)", name="ck_adult_age"),
        CheckConstraint("male IN (0, 1)", name="ck_adult_male"),
        CheckConstraint("internet IN (0, 1)", name="ck_adult_internet"),
        CheckConstraint("pc_internet IN (0, 1)", name="ck_adult_pc_internet"),
    )

    pid: Mapped[int] = mapped_column(Integer, primary_key=True)
    age: Mapped[float | None] = mapped_column(Float, nullable=True)
    male: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    edu_year: Mapped[float | None] = mapped_column(Float, nullable=True)
    internet: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pc_internet: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Household(Base):
    __tablename__ = "households"
    __table_args__ = (
        CheckConstraint("service_adopt IN (0, 1)", name="ck_household_adopt"),
        CheckConstraint("familysize IS NULL OR familysize > 0", name="ck_household_size"),
    )

    fid22: Mapped[int] = mapped_column(Integer, primary_key=True)
    resp1pid: Mapped[int] = mapped_column(
        ForeignKey("adults.pid"), nullable=False, index=True
    )
    familysize: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ln_income_per: Mapped[float | None] = mapped_column(Float, nullable=True)
    service_adopt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    land_asset_log: Mapped[float | None] = mapped_column(Float, nullable=True)


class AppMeta(Base):
    __tablename__ = "app_meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(255), nullable=False)
