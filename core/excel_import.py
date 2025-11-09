"""Utilities to import member timeseries data from Excel workbooks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from typing import Literal, List
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


class ExcelTimeseriesFormatError(ValueError):
    """Raised when an Excel file cannot be converted into a usable timeseries."""


DatasetType = Literal["consumption", "production"]


@dataclass(frozen=True)
class ExcelTimeseriesResult:
    """Represents the output of an Excel to CSV conversion."""

    csv_content: str
    annual_consumption_kwh: float
    annual_production_kwh: float


def _select_numeric_series(frame: pd.DataFrame) -> pd.Series:
    """Extract the numeric series with the largest absolute total from the dataframe."""

    numeric = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if numeric.empty:
        raise ExcelTimeseriesFormatError("No numeric data detected in Excel sheet.")

    column_totals = numeric.abs().sum()
    if column_totals.max() <= 0:
        raise ExcelTimeseriesFormatError("Numeric columns do not contain positive values.")

    selected_column = column_totals.idxmax()
    return numeric[selected_column]


def _is_numeric(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _column_ref_to_index(cell_reference: str) -> int:
    letters = ""
    for char in cell_reference:
        if char.isalpha():
            letters += char.upper()
        else:
            break
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1 if letters else 0


def _load_xlsx_to_dataframe(raw_bytes: bytes) -> pd.DataFrame:
    ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ns_pkg = "http://schemas.openxmlformats.org/package/2006/relationships"

    with zipfile.ZipFile(BytesIO(raw_bytes)) as archive:
        try:
            workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        except KeyError as exc:
            raise ExcelTimeseriesFormatError("Workbook is missing core metadata.") from exc

        sheets_parent = workbook_root.find(f"{{{ns_main}}}sheets")
        if sheets_parent is None or not list(sheets_parent):
            raise ExcelTimeseriesFormatError("Workbook does not contain any sheets.")

        first_sheet = list(sheets_parent)[0]
        rel_id = first_sheet.attrib.get(f"{{{ns_rel}}}id")
        if not rel_id:
            raise ExcelTimeseriesFormatError("Unable to locate sheet relationship identifier.")

        try:
            relationships_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        except KeyError as exc:
            raise ExcelTimeseriesFormatError("Workbook relationships are missing.") from exc

        target = None
        for relationship in relationships_root.findall(f"{{{ns_pkg}}}Relationship"):
            if relationship.attrib.get("Id") == rel_id:
                target = relationship.attrib.get("Target")
                break
        if not target:
            raise ExcelTimeseriesFormatError("Unable to resolve worksheet file in workbook.")

        worksheet_path = target if target.startswith("xl/") else f"xl/{target}"
        try:
            sheet_root = ET.fromstring(archive.read(worksheet_path))
        except KeyError as exc:
            raise ExcelTimeseriesFormatError("Worksheet file referenced in workbook is missing.") from exc

        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for element in shared_root.findall(f".//{{{ns_main}}}si"):
                text = "".join(t.text or "" for t in element.findall(f".//{{{ns_main}}}t"))
                shared_strings.append(text)

        rows: List[List[str]] = []
        max_columns = 0
        for row in sheet_root.findall(f".//{{{ns_main}}}row"):
            row_values: List[str] = []
            expected_index = 0
            for cell in row.findall(f"{{{ns_main}}}c"):
                ref = cell.attrib.get("r", "")
                column_index = _column_ref_to_index(ref)
                while expected_index < column_index:
                    row_values.append("")
                    expected_index += 1

                cell_type = cell.attrib.get("t")
                value_element = cell.find(f"{{{ns_main}}}v")
                value_text = value_element.text if value_element is not None else ""

                if cell_type == "s" and value_text:
                    try:
                        value_text = shared_strings[int(value_text)]
                    except (IndexError, ValueError):
                        value_text = ""

                row_values.append(value_text)
                expected_index += 1

            rows.append(row_values)
            max_columns = max(max_columns, len(row_values))

        if not rows:
            raise ExcelTimeseriesFormatError("Worksheet does not contain any rows.")

        for row_values in rows:
            if len(row_values) < max_columns:
                row_values.extend([""] * (max_columns - len(row_values)))

        first_row = rows[0]
        if any(not _is_numeric(value) for value in first_row if value != ""):
            headers = first_row
            data_rows = rows[1:]
        else:
            headers = [f"Column {idx}" for idx in range(max_columns)]
            data_rows = rows

        frame = pd.DataFrame(data_rows, columns=headers)
        return frame


def convert_excel_timeseries_to_csv(uploaded_file, dataset_type: DatasetType) -> ExcelTimeseriesResult:
    """Convert a 15-minute Excel timeseries into the internal CSV representation."""

    dataset_type = dataset_type or "consumption"
    if dataset_type not in ("consumption", "production"):
        raise ExcelTimeseriesFormatError("Unsupported dataset type.")

    raw_bytes = uploaded_file.read()
    if not raw_bytes:
        raise ExcelTimeseriesFormatError("Empty Excel file provided.")

    uploaded_file.seek(0)

    try:
        dataframe = _load_xlsx_to_dataframe(raw_bytes)
    except ExcelTimeseriesFormatError:
        raise
    except Exception as exc:
        raise ExcelTimeseriesFormatError(f"Unable to read Excel workbook: {exc}") from exc

    dataframe = dataframe.dropna(how="all")
    if dataframe.shape[0] < 96:
        raise ExcelTimeseriesFormatError("Excel sheet must contain at least 96 rows of data.")

    if dataframe.shape[1] < 2:
        raise ExcelTimeseriesFormatError("Excel sheet must contain at least two columns (time and values).")

    # Skip the first column, assumed to be timestamps, and keep the numerical series.
    numeric_frame = dataframe.iloc[:, 1:]
    values_series = _select_numeric_series(numeric_frame)

    values = values_series.to_numpy(dtype=float)
    if values.ndim != 1:
        values = values.reshape(-1)

    total_points = len(values)
    full_days = total_points // 96
    if full_days == 0:
        raise ExcelTimeseriesFormatError("Excel sheet does not provide a full set of 96-step days.")

    trimmed_points = full_days * 96
    values = values[:trimmed_points]

    matrix = values.reshape(full_days, 96)
    profile = matrix.mean(axis=0)

    # Numerical totals for Stage 3 calculations.
    total_energy = float(np.maximum(values, 0.0).sum())
    annual_consumption = total_energy if dataset_type == "consumption" else 0.0
    annual_production = total_energy if dataset_type == "production" else 0.0

    timestamps = []
    start = datetime(2000, 1, 1, 0, 0)
    for slot in range(96):
        timestamps.append((start + timedelta(minutes=15 * slot)).strftime("%H:%M"))

    production_profile = profile if dataset_type == "production" else np.zeros_like(profile)
    consumption_profile = profile if dataset_type == "consumption" else np.zeros_like(profile)

    buffer = StringIO()
    buffer.write("Time,Production,Consommation\n")
    for idx in range(96):
        buffer.write(
            f"{timestamps[idx]},{production_profile[idx]:.6f},{consumption_profile[idx]:.6f}\n"
        )

    return ExcelTimeseriesResult(
        csv_content=buffer.getvalue(),
        annual_consumption_kwh=annual_consumption,
        annual_production_kwh=annual_production,
    )
