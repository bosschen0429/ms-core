"""Excel 格式複製工具函式。

提供工作表格式複製功能，用於在 pandas 寫入 Excel 後保留原始格式
（如 ISTD 紅色標記、border、alignment、number_format 等）。
"""

from copy import copy


def copy_cell_style(src_cell, tgt_cell):
    """複製單一儲存格的所有樣式（font, border, fill, number_format, protection, alignment）"""
    if src_cell.has_style:
        tgt_cell.font = copy(src_cell.font)
        tgt_cell.border = copy(src_cell.border)
        tgt_cell.fill = copy(src_cell.fill)
        tgt_cell.number_format = src_cell.number_format
        tgt_cell.protection = copy(src_cell.protection)
        tgt_cell.alignment = copy(src_cell.alignment)


def copy_sheet_with_style(src_ws, tgt_ws):
    """複製工作表的所有儲存格值與格式，含列寬、行高、合併儲存格。"""
    for row in src_ws.iter_rows():
        for cell in row:
            new_cell = tgt_ws.cell(row=cell.row, column=cell.column, value=cell.value)
            copy_cell_style(cell, new_cell)

    # 列寬
    for col_letter, col_dim in src_ws.column_dimensions.items():
        tgt_ws.column_dimensions[col_letter].width = col_dim.width

    # 行高
    for row_num, row_dim in src_ws.row_dimensions.items():
        tgt_ws.row_dimensions[row_num].height = row_dim.height

    # 合併儲存格
    for merged_range in src_ws.merged_cells.ranges:
        tgt_ws.merge_cells(str(merged_range))


def copy_sheet_formatting_only(src_ws, tgt_ws):
    """只複製格式（不覆蓋值），用於 pandas 寫入後補回格式。

    以 src_ws 的 max_row/max_column 為範圍，逐格複製格式到 tgt_ws 的相同位置。
    適用於：pandas 已寫入正確的值，但格式（紅色 ISTD 標記等）遺失的情況。
    """
    for row in range(1, src_ws.max_row + 1):
        for col_idx in range(1, src_ws.max_column + 1):
            src_cell = src_ws.cell(row=row, column=col_idx)
            if src_cell.has_style:
                tgt_cell = tgt_ws.cell(row=row, column=col_idx)
                copy_cell_style(src_cell, tgt_cell)
