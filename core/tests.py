from io import BytesIO
import zipfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

import numpy as np

from .excel_import import convert_excel_timeseries_to_csv, ExcelTimeseriesFormatError


class ExcelImportTests(TestCase):
    def _column_letter(self, index: int) -> str:
        letters = ""
        index += 1
        while index:
            index, remainder = divmod(index - 1, 26)
            letters = chr(65 + remainder) + letters
        return letters

    def _build_excel_file(self, values, column_count=2):
        rows = []
        header = ["Date"] + [f"Valeur{i}" for i in range(1, column_count)]
        rows.append(header)

        for idx, value in enumerate(values, start=1):
            row = [str(idx)]
            for multiplier in range(1, column_count):
                row.append(str(float(value) * multiplier))
            rows.append(row)

        sheet_rows = []
        for row_idx, row_values in enumerate(rows, start=1):
            cells = []
            for col_idx, cell_value in enumerate(row_values):
                column_letter = self._column_letter(col_idx)
                cell_ref = f"{column_letter}{row_idx}"
                if row_idx == 1:
                    cells.append(f'<c r="{cell_ref}" t="str"><v>{cell_value}</v></c>')
                else:
                    cells.append(f'<c r="{cell_ref}"><v>{cell_value}</v></c>')
            sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

        sheet_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<sheetData>{''.join(sheet_rows)}</sheetData>"
            '</worksheet>'
        )

        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>'
        )

        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            '</Relationships>'
        )

        root_rels = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '</Relationships>'
        )

        content_types = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '</Types>'
        )

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", root_rels)
            archive.writestr("xl/workbook.xml", workbook_xml)
            archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
            archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)

        buffer.seek(0)

        return SimpleUploadedFile(
            "test.xlsx",
            buffer.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    def test_consumption_excel_conversion(self):
        values = np.ones(192)  # 2 days of hourly data at 15-minute resolution
        excel_file = self._build_excel_file(values)

        result = convert_excel_timeseries_to_csv(excel_file, "consumption")

        self.assertAlmostEqual(result.annual_consumption_kwh, 192.0)
        self.assertEqual(result.annual_production_kwh, 0.0)

        lines = result.csv_content.strip().splitlines()
        self.assertEqual(len(lines), 97)  # header + 96 values

    def test_production_excel_conversion(self):
        values = np.linspace(0, 10, 96)
        excel_file = self._build_excel_file(values, column_count=3)

        result = convert_excel_timeseries_to_csv(excel_file, "production")

        self.assertAlmostEqual(result.annual_production_kwh, float(values.sum() * 2), places=5)
        self.assertEqual(result.annual_consumption_kwh, 0.0)

    def test_invalid_dataset_type(self):
        values = np.ones(96)
        excel_file = self._build_excel_file(values)

        with self.assertRaises(ExcelTimeseriesFormatError):
            convert_excel_timeseries_to_csv(excel_file, "invalid")
