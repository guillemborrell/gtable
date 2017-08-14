from gtable.lib import merge_table
from gtable import Table


def merge(table_left, table_right, column):
    data, keys, index = merge_table(table_left, table_right, column)
    table = Table()
    table.data = data
    table.keys = keys
    table.index = index

    return table

