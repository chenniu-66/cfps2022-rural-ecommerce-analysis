from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session

from .models import Adult, Household
from .schemas import (
    AdultOut,
    GroupMetric,
    GroupResponse,
    HouseholdDetail,
    HouseholdOut,
    OverviewResponse,
    ValidationCheck,
    ValidationReport,
)


DIMENSIONS = {"internet", "pc_internet", "gender", "age_group", "education", "income_quartile"}


def get_overview(db: Session) -> OverviewResponse:
    households = db.scalar(select(func.count()).select_from(Household)) or 0
    adults = db.scalar(select(func.count()).select_from(Adult)) or 0
    matched = db.scalar(
        select(func.count())
        .select_from(Household)
        .join(Adult, Household.resp1pid == Adult.pid)
    ) or 0
    adopted = db.scalar(select(func.coalesce(func.sum(Household.service_adopt), 0))) or 0
    rate = float(adopted) / households if households else 0.0
    return OverviewResponse(
        households=households,
        adults=adults,
        matched_pairs=matched,
        adopted_households=int(adopted),
        adoption_rate=round(rate, 6),
    )


def _group_expression(dimension: str):
    if dimension == "internet":
        return case((Adult.internet == 1, "上网"), else_="未上网")
    if dimension == "pc_internet":
        return case((Adult.pc_internet == 1, "电脑上网"), else_="仅移动上网")
    if dimension == "gender":
        return case((Adult.male == 1, "男性"), else_="女性")
    if dimension == "age_group":
        return case(
            (Adult.age < 30, "18–29 岁"),
            (Adult.age < 40, "30–39 岁"),
            (Adult.age < 50, "40–49 岁"),
            (Adult.age < 60, "50–59 岁"),
            else_="60 岁及以上",
        )
    if dimension == "education":
        return case(
            (Adult.edu_year <= 6, "小学及以下"),
            (Adult.edu_year <= 9, "初中"),
            (Adult.edu_year <= 12, "高中/中专"),
            (Adult.edu_year <= 15, "大专"),
            else_="本科及以上",
        )
    raise ValueError(f"Unsupported dimension: {dimension}")


def get_group_metrics(db: Session, dimension: str) -> GroupResponse:
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {sorted(DIMENSIONS)}")

    if dimension == "income_quartile":
        ranked = (
            select(
                Household.service_adopt.label("y"),
                func.ntile(4).over(order_by=Household.ln_income_per).label("quartile"),
            )
            .where(Household.ln_income_per.is_not(None))
            .subquery()
        )
        stmt = (
            select(
                ranked.c.quartile,
                func.count().label("n"),
                func.coalesce(func.sum(ranked.c.y), 0).label("adopted"),
            )
            .group_by(ranked.c.quartile)
            .order_by(ranked.c.quartile)
        )
        rows = db.execute(stmt).all()
        groups = [
            GroupMetric(
                dimension=dimension,
                group=f"收入 Q{int(row.quartile)}",
                n=int(row.n),
                adopted=int(row.adopted),
                adoption_rate=round(float(row.adopted) / int(row.n), 6) if row.n else 0.0,
            )
            for row in rows
        ]
        return GroupResponse(dimension=dimension, groups=groups)

    label = _group_expression(dimension)
    stmt = (
        select(
            label.label("group_label"),
            func.count().label("n"),
            func.coalesce(func.sum(Household.service_adopt), 0).label("adopted"),
        )
        .select_from(Household)
        .join(Adult, Household.resp1pid == Adult.pid)
        .where(label.is_not(None))
        .group_by(label)
        .order_by(label)
    )
    rows = db.execute(stmt).all()
    groups = [
        GroupMetric(
            dimension=dimension,
            group=str(row.group_label),
            n=int(row.n),
            adopted=int(row.adopted),
            adoption_rate=round(float(row.adopted) / int(row.n), 6) if row.n else 0.0,
        )
        for row in rows
    ]
    return GroupResponse(dimension=dimension, groups=groups)


def get_household(db: Session, fid22: int) -> HouseholdDetail | None:
    row = db.execute(
        select(Household, Adult)
        .join(Adult, Household.resp1pid == Adult.pid)
        .where(Household.fid22 == fid22)
    ).first()
    if row is None:
        return None
    household, adult = row
    return HouseholdDetail(
        household=HouseholdOut.model_validate(household),
        adult=AdultOut.model_validate(adult),
    )


def get_validation_report(db: Session) -> ValidationReport:
    checks: list[ValidationCheck] = []

    def add(name: str, ok: bool, value: int | float, expected: str) -> None:
        checks.append(
            ValidationCheck(
                name=name,
                status="pass" if ok else "fail",
                value=value,
                expected=expected,
            )
        )

    households = db.scalar(select(func.count()).select_from(Household)) or 0
    distinct_households = db.scalar(select(func.count(func.distinct(Household.fid22)))) or 0
    adults = db.scalar(select(func.count()).select_from(Adult)) or 0
    distinct_adults = db.scalar(select(func.count(func.distinct(Adult.pid)))) or 0

    duplicate_households = households - distinct_households
    duplicate_adults = adults - distinct_adults
    unmatched = db.scalar(
        select(func.count())
        .select_from(Household)
        .outerjoin(Adult, Household.resp1pid == Adult.pid)
        .where(Adult.pid.is_(None))
    ) or 0
    invalid_adoption = db.scalar(
        select(func.count())
        .select_from(Household)
        .where(or_(Household.service_adopt.is_(None), Household.service_adopt.notin_([0, 1])))
    ) or 0
    missing_age_or_edu = db.scalar(
        select(func.count())
        .select_from(Adult)
        .where(or_(Adult.age.is_(None), Adult.edu_year.is_(None)))
    ) or 0

    add("household_unique_key", duplicate_households == 0, duplicate_households, "0 duplicate fid22")
    add("adult_unique_key", duplicate_adults == 0, duplicate_adults, "0 duplicate pid")
    add("household_adult_join", unmatched == 0, unmatched, "0 unmatched households")
    add("service_adopt_domain", invalid_adoption == 0, invalid_adoption, "all values in {0, 1}")
    add("age_education_complete", missing_age_or_edu == 0, missing_age_or_edu, "0 missing age/edu")

    overview = get_overview(db)
    add("adoption_rate_range", 0 <= overview.adoption_rate <= 1, overview.adoption_rate, "0 <= rate <= 1")

    status = "pass" if all(c.status == "pass" for c in checks) else "fail"
    return ValidationReport(
        status=status,
        generated_at=datetime.now(timezone.utc),
        checks=checks,
    )
