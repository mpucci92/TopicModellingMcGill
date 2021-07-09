from bokeh.embed import components
from bokeh.models import ColumnDataSource, TableColumn, DataTable

def tableBokeh(topic_df):

    source = ColumnDataSource(topic_df)

    columns = [
        TableColumn(field='topic', title='topic'),
        TableColumn(field='keywords', title='keywords'),
        TableColumn(field='score', title='score')
    ]

    myTable = DataTable(source=source, columns=columns)
    script, div = components(myTable)

    return script,div