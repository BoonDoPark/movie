"""
/***************************************************************************
        begin                : 2021-04-15
        email                : hsk@git.co.kr
 ***************************************************************************/
"""
from abc import *
from collections import OrderedDict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidget
from PyQt5.QtWidgets import QTableWidgetItem

from ui_utils_checkbox import QCheckBoxUtils


class QTableFormat:
    def __init__(self):
        self._form = OrderedDict()

    def __len__(self):
        return len(self._form)

    def __sub__(self, row: tuple):
        self._form.pop(row)

    def __getitem__(self, key):
        if key in self._form.keys():
            return self._form[key]
        else:
            raise KeyError

    @property
    def form(self):
        return self._form

    def append_by_row(self, row_data: tuple):
        self._form[len(self._form)] = row_data

    def edit_row(self, row: int, row_data: tuple):
        try:
            self._form[row] = row_data
        except KeyError:
            pass

    def delete_row(self, row: int):
        self._form.pop(row)

    def get(self, row: int):
        return self._form.get(row)

    def items(self):
        for r, row_data in self._form.items():
            yield r, row_data

    def keys(self):
        for r in self._form.keys():
            yield r

    def values(self):
        for row_data in self._form.values():
            yield row_data

    def clear(self):
        self._form.clear()


class QTableWidgetUtils:
    padding = 30

    @staticmethod
    def resize_table_widget(table_widget: QTableWidget, resize_columns=True, resize_rows=True):
        if resize_columns:
            for c in range(table_widget.columnCount()):
                table_widget.resizeColumnToContents(c)
                width = table_widget.columnWidth(c)
                table_widget.setColumnWidth(c, width + QTableWidgetUtils.padding)

        if resize_rows:
            for row in range(table_widget.rowCount()):
                table_widget.resizeRowToContents(row)
            # table_widget.resizeRowsToContents()

    @staticmethod
    def set_true_check_boxes_in_column(table_widget: QTableWidget, column: int):
        """
        2021.03.26.hsk : 오류 딕셔너리 확인후 체크박스 체크
        :return:
        """
        for r in range(table_widget.rowCount()):
            widget, checkbox = QCheckBoxUtils.check_box_aligned_center()
            table_widget.setCellWidget(r, column, widget)
            checkbox.setCheckable(True)
            checkbox.setChecked(True)

    @staticmethod
    def set_false_check_boxes_in_column(table_widget: QTableWidget, column: int):
        """
        2021.03.26.hsk : 오류 딕셔너리 확인후 체크박스 체크
        :return:
        """
        for r in range(table_widget.rowCount()):
            widget, checkbox = QCheckBoxUtils.check_box_aligned_center()
            table_widget.setCellWidget(r, column, widget)
            checkbox.setCheckable(True)
            checkbox.setChecked(False)

    @classmethod
    def refresh_by_items(cls, table_widget: QTableWidget, display_data: QTableFormat, user_data: QTableFormat = None):
        """
        2021.04.15.hsk : QTableWidget 갱신
        :param table_widget: QTableWidget
        :param display_data: QTableFormat
        ex)
        [0: (1, 2, 3),
         1: (4, 5, 6),
         2: (7, 8, 9)]
        :param user_data: QTableFormat
        display_data 사이즈보다 커야한다.
        채우지 않을 부분은 None 으로 매꾼다.
        ex)
        [0: (None, None, 1),
         1: (None, None, 2),
         2: (None, None, 3)]
        """
        table_widget.setRowCount(0)
        for r, row_datum in display_data.items():
            table_widget.insertRow(r)
            for c, column_datum in enumerate(row_datum):
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
                column_datum = ' ' if column_datum == '' else column_datum
                if column_datum and cls._is_string(column_datum):
                    item.setData(Qt.DisplayRole, column_datum)
                    if user_data is not None:
                        if user_data.get(r)[c]:
                            item.setData(Qt.UserRole, user_data.get(r)[c])
                table_widget.setItem(r, c, item)

    @staticmethod
    def _is_string(num):
        try:
            str(num)
            return True
        except ValueError:
            return False
