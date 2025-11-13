"""Utility helpers to parse and validate quarter-hour time series files."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile


class TimeseriesError(Exception):
    """Raised when an uploaded file cannot be parsed as a time series."""


@dataclass
class TimeseriesMetadata:
    row_count: int
    start: Optional[str]
    end: Optional[str]
    granularity_minutes: Optional[float]
    coverage_days: Optional[float]
    missing_rows: Optional[int]
    totals: Dict[str, float]
    detected_columns: Dict[str, Optional[str]]
    file_type: str
    normalized_year: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class TimeseriesResult:
    data: pd.DataFrame
    metadata: TimeseriesMetadata


Readable = Union[str, Path, UploadedFile]

PRODUCTION_KEYWORDS = ("prod", "export", "injection", "generation")
CONSUMPTION_KEYWORDS = ("conso", "load", "a+", "import", "consumption")


def parse_member_timeseries(source: Readable) -> TimeseriesResult:
    """Parse a member time series file.

    Returns production/consumption in kWh with a metadata summary.
    """

    frame, timestamp_col = _load_base_dataframe(source)

    numeric_cols = [c for c in frame.columns if c != timestamp_col]
    if not numeric_cols:
        raise TimeseriesError("Le fichier ne contient aucune colonne numérique exploitable.")

    production_col, consumption_col = _identify_energy_columns(numeric_cols)

    detected_cols: Dict[str, Optional[str]] = {
        "timestamp": timestamp_col,
        "production": production_col,
        "consumption": consumption_col,
    }
    warnings: List[str] = []

    df = frame[[timestamp_col]].rename(columns={timestamp_col: "timestamp"})
    df["timestamp"], normalized_year = _normalise_to_reference_year(df["timestamp"])

    if production_col:
        df["production_kwh"] = _to_float_series(frame[production_col])
    else:
        df["production_kwh"] = 0.0

    if consumption_col:
        df["consumption_kwh"] = _to_float_series(frame[consumption_col])
    else:
        df["consumption_kwh"] = 0.0

    if not production_col and not consumption_col:
        # Fallback: use the first numeric column as consumption.
        fallback = numeric_cols[0]
        df["consumption_kwh"] = _to_float_series(frame[fallback])
        detected_cols["consumption"] = fallback
        warnings.append("Aucune colonne Production/Consommation détectée automatiquement. La première colonne numérique a été utilisée comme consommation.")

    metadata = build_metadata(
        df,
        detected_cols,
        file_type="member_timeseries",
        warnings=warnings,
        normalized_year=normalized_year,
    )
    return TimeseriesResult(df, metadata)


def parse_profile_timeseries(source: Readable, profile_type: str) -> TimeseriesResult:
    """Parse a profile (consumption or production) file normalised on 1 MWh."""

    frame, timestamp_col = _load_base_dataframe(source)

    numeric_cols = [c for c in frame.columns if c != timestamp_col]
    if not numeric_cols:
        raise TimeseriesError("Le fichier de profil ne contient aucune colonne numérique.")

    preferred = _select_profile_column(profile_type, numeric_cols)
    df = frame[[timestamp_col, preferred]].rename(
        columns={timestamp_col: "timestamp", preferred: "value_kwh"}
    )
    df["timestamp"], normalized_year = _normalise_to_reference_year(df["timestamp"])
    df["value_kwh"] = _to_float_series(df["value_kwh"])

    value_total = float(df["value_kwh"].sum())
    if profile_type == "production":
        metadata_frame = df.assign(production_kwh=df["value_kwh"], consumption_kwh=0.0)
        totals = {
            "value_kwh": value_total,
            "production_kwh": value_total,
            "consumption_kwh": 0.0,
        }
    else:
        metadata_frame = df.assign(production_kwh=0.0, consumption_kwh=df["value_kwh"])
        totals = {
            "value_kwh": value_total,
            "production_kwh": 0.0,
            "consumption_kwh": value_total,
        }

    metadata = build_metadata(
        metadata_frame,
        {"timestamp": timestamp_col, "value": preferred},
        file_type=f"profile_{profile_type}",
        totals_override=totals,
        normalized_year=normalized_year,
    )

    return TimeseriesResult(df, metadata)


def _load_base_dataframe(source: Readable) -> Tuple[pd.DataFrame, str]:
    df = _read_tabular(source)
    if df.empty:
        raise TimeseriesError("Le fichier est vide.")

    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    timestamp_col = _detect_timestamp_column(df.columns, df)
    if not timestamp_col:
        raise TimeseriesError(
            "Impossible d'identifier la colonne temporelle. Assurez-vous que la première colonne contient des dates." 
        )

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col)

    return df, timestamp_col


def _read_tabular(source: Readable) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
            return pd.read_excel(path)
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path, sep=";", engine="python")

    # Uploaded file like Django's InMemoryUploadedFile
    upload = source
    name = (upload.name or "").lower()
    upload.seek(0)
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(upload)
        upload.seek(0)
        return df

    try:
        df = pd.read_csv(upload, sep=None, engine="python")
    except Exception:
        upload.seek(0)
        df = pd.read_csv(upload, sep=";", engine="python")
    upload.seek(0)
    return df


def _detect_timestamp_column(columns: Iterable[str], df: pd.DataFrame) -> Optional[str]:
    for col in columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        except Exception:
            continue
        if parsed.notna().mean() > 0.6:
            return col
    return None


def _identify_energy_columns(columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
    production = None
    consumption = None
    for col in columns:
        name = col.lower()
        if production is None and any(keyword in name for keyword in PRODUCTION_KEYWORDS):
            production = col
            continue
        if consumption is None and any(keyword in name for keyword in CONSUMPTION_KEYWORDS):
            consumption = col
            continue

    # Additional heuristic: if one column looks like an MWh duplicate (contains 1000), prefer the other one.
    if consumption is None:
        candidates = [c for c in columns if "1000" not in c.lower() and "mwh" not in c.lower()]
        if candidates:
            consumption = candidates[0]
    if production is None:
        remaining = [c for c in columns if c != consumption]
        if remaining:
            production = remaining[0]

    return production, consumption


def _select_profile_column(profile_type: str, columns: List[str]) -> str:
    lowered = {col: col.lower() for col in columns}

    def _matches(col: str, keywords: Iterable[str]) -> bool:
        return any(keyword in lowered[col] for keyword in keywords)

    thousand_cols = [
        col for col in columns if "1000" in lowered[col] or "mwh" in lowered[col]
    ]
    base_cols = [col for col in columns if col not in thousand_cols]

    if profile_type == "consumption":
        preference_groups = [
            [col for col in thousand_cols if _matches(col, CONSUMPTION_KEYWORDS)],
            thousand_cols,
            [col for col in base_cols if _matches(col, CONSUMPTION_KEYWORDS)],
            base_cols,
        ]
    elif profile_type == "production":
        preference_groups = [
            [col for col in base_cols if _matches(col, PRODUCTION_KEYWORDS)],
            [col for col in columns if _matches(col, PRODUCTION_KEYWORDS)],
            base_cols,
        ]
    else:
        preference_groups = [
            [
                col
                for col in base_cols
                if _matches(col, PRODUCTION_KEYWORDS + CONSUMPTION_KEYWORDS)
            ],
            base_cols,
            thousand_cols,
        ]

    for group in preference_groups:
        if group:
            return group[0]

    return columns[0]


def _to_float_series(series: pd.Series) -> pd.Series:
    """Return a float series tolerant to common European number formats."""

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0)

    text = (
        series.astype(str)
        .str.strip()
        .str.replace("\u00a0", "", regex=False)  # non-breaking space
        .str.replace(r"\s+", "", regex=True)
    )

    # When both comma and dot are present we assume the dot is a thousands separator.
    both_sep_mask = text.str.contains(",") & text.str.contains(".", regex=False)
    text = text.where(~both_sep_mask, text.str.replace(".", "", regex=False))

    # Apostrophes are also used as thousands separators in some exports.
    text = text.str.replace("'", "", regex=False)

    normalized = text.str.replace(",", ".", regex=False)

    result = pd.to_numeric(normalized, errors="coerce")
    return result.fillna(0.0)


def build_metadata(
    df: pd.DataFrame,
    detected: Dict[str, Optional[str]],
    *,
    file_type: str,
    totals_override: Optional[Dict[str, float]] = None,
    warnings: Optional[List[str]] = None,
    normalized_year: Optional[int] = None,
) -> TimeseriesMetadata:
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    timestamps = timestamps.dropna()
    row_count = len(df)

    start_ts = timestamps.min()
    end_ts = timestamps.max()

    granularity = None
    coverage_days = None
    missing_rows = None
    granularity_minutes: Optional[float] = None

    if len(timestamps) > 1:
        diffs = timestamps.diff().dropna()
        if not diffs.empty:
            granularity = diffs.mode().iloc[0]
            if isinstance(granularity, pd.Timedelta):
                granularity_minutes = granularity / timedelta(minutes=1)
            else:
                granularity_minutes = float(granularity) / timedelta(minutes=1)
            granularity_minutes = float(granularity_minutes)

    if start_ts is not None and end_ts is not None and granularity_minutes:
        total_minutes = (end_ts - start_ts).total_seconds() / 60
        if granularity_minutes:
            expected_rows = int(round(total_minutes / granularity_minutes)) + 1
            missing_rows = max(expected_rows - row_count, 0)
        coverage_days = total_minutes / (60 * 24)

    totals = totals_override or {
        "production_kwh": float(df.get("production_kwh", pd.Series(dtype=float)).sum()),
        "consumption_kwh": float(df.get("consumption_kwh", pd.Series(dtype=float)).sum()),
    }

    metadata = TimeseriesMetadata(
        row_count=row_count,
        start=start_ts.isoformat() if pd.notna(start_ts) else None,
        end=end_ts.isoformat() if pd.notna(end_ts) else None,
        granularity_minutes=granularity_minutes,
        coverage_days=coverage_days,
        missing_rows=missing_rows,
        totals=totals,
        detected_columns=detected,
        file_type=file_type,
        normalized_year=normalized_year or (int(start_ts.year) if pd.notna(start_ts) else None),
        warnings=warnings or [],
    )
    return metadata


def _get_reference_year() -> Optional[int]:
    year = getattr(settings, "TIMESERIES_REFERENCE_YEAR", 2025)
    if year in (None, "", False):
        return None
    try:
        value = int(year)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _next_leap_year(year: int) -> int:
    candidate = year
    while not _is_leap_year(candidate):
        candidate += 1
    return candidate


def _normalise_to_reference_year(series: pd.Series) -> Tuple[pd.Series, Optional[int]]:
    timestamps = pd.to_datetime(series, errors="coerce")
    normalized = pd.Series(timestamps, index=series.index)

    reference_year = _get_reference_year()
    if reference_year is None:
        return normalized, None

    valid = normalized.dropna().sort_values()
    if valid.empty:
        return normalized, None

    start = valid.iloc[0]
    target_year = reference_year
    contains_feb_29 = ((valid.dt.month == 2) & (valid.dt.day == 29)).any()
    if contains_feb_29 and not _is_leap_year(target_year):
        target_year = _next_leap_year(target_year)

    try:
        target_start = start.replace(year=target_year)
    except ValueError:
        # Should not happen thanks to leap-year adjustment, but keep a fallback.
        target_start = datetime(target_year, start.month, min(start.day, 28), start.hour, start.minute, start.second)

    first_value = start
    for idx, original in normalized.dropna().items():
        delta = original - first_value
        normalized.at[idx] = target_start + delta

    return normalized, target_year
