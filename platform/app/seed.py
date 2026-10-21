from __future__ import annotations

import math
import random
from datetime import datetime, timezone

from sqlalchemy import func, select

from .database import Base, SessionLocal, engine
from .models import Adult, AppMeta, Household


DEMO_SIZE = 1500
RNG = random.Random(2027)


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _make_row(index: int) -> tuple[Adult, Household]:
    fid22 = 100_001 + index
    pid = fid22 * 100 + 1

    age = _clamp(RNG.gauss(45, 14), 18, 80)
    male = RNG.choices([0, 1], weights=[0.48, 0.52])[0]
    edu = _clamp(RNG.gauss(8.5 + 0.035 * (age - 40), 3.2), 0, 19)
    family_size = RNG.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[8, 18, 24, 22, 14, 8, 4, 2])[0]

    ln_income = RNG.gauss(9.7 + 0.08 * edu, 0.85)
    ln_income = _clamp(ln_income, 6.5, 13.5)

    internet_p = _sigmoid(-0.7 + 0.09 * (age - 40) - 0.0025 * (age - 40) ** 2 + 0.13 * edu)
    internet = 1 if RNG.random() < internet_p else 0

    pc_p = _sigmoid(-2.1 + 0.12 * edu + 0.5 * internet - 0.02 * age)
    pc_internet = 1 if internet and RNG.random() < pc_p else 0

    adopt_p = _sigmoid(
        -1.0
        + 0.9 * internet
        + 0.06 * edu
        + 0.24 * (ln_income - 9.5)
        - 0.012 * (age - 40)
        - 0.0007 * (age - 40) ** 2
        + 0.35 * pc_internet
        + 0.10 * male
    )
    service_adopt = 1 if RNG.random() < adopt_p else 0

    land_asset_log = None
    if RNG.random() > 0.15:
        land_asset_log = _clamp(RNG.gauss(7.0, 2.0), 0, 15)

    adult = Adult(
        pid=pid,
        age=round(age, 1),
        male=male,
        edu_year=round(edu, 1),
        internet=internet,
        pc_internet=pc_internet,
    )
    household = Household(
        fid22=fid22,
        resp1pid=pid,
        familysize=family_size,
        ln_income_per=round(ln_income, 4),
        service_adopt=service_adopt,
        land_asset_log=round(land_asset_log, 4) if land_asset_log is not None else None,
    )
    return adult, household


def seed_if_empty(db) -> int:
    """Populate synthetic demo rows when the database is empty.

    The rows are deliberately synthetic and must never be presented as CFPS
    observations. They exist so the API can be started and tested without
    distributing the licensed CFPS microdata.
    """
    existing = db.scalar(select(func.count()).select_from(Household)) or 0
    if existing:
        return 0

    adults: list[Adult] = []
    households: list[Household] = []
    for i in range(DEMO_SIZE):
        adult, household = _make_row(i)
        adults.append(adult)
        households.append(household)

    db.add_all(adults)
    db.flush()
    db.add_all(households)
    db.add_all([
        AppMeta(key="data_mode", value="demo"),
        AppMeta(key="seeded_at", value=datetime.now(timezone.utc).isoformat()),
        AppMeta(key="seed_size", value=str(DEMO_SIZE)),
        AppMeta(
            key="data_notice",
            value="Synthetic demo data only. CFPS microdata is not included.",
        ),
    ])
    db.commit()
    return DEMO_SIZE


def init_database() -> int:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        return seed_if_empty(db)


if __name__ == "__main__":
    created = init_database()
    print(f"database ready; seeded {created} synthetic households")
