from bokeh.resources import INLINE
from bokeh.embed import components
from main import *
from datetime import date
from flask import Flask, render_template
from topicDataframe import *
from waitress import serve
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.io import show


def circleBokeh(result,data):

    palette = d3['Category20'][20][:15]
    grey = palette[-1]
    palette = palette[:-1]

    result['title'] = data

    result['color'] = [
        palette[label % len(palette)]
        for label in result.labels.values
    ]

    result.loc[result.labels == -1, 'color'] = grey
    result = result[result.labels != -1]

    source = bpl.ColumnDataSource(data=result)
    hover = HoverTool(tooltips=[('title', '@title'), ('topic', '@labels')])

    # title for the graph

    p = bpl.figure(title="Topic Modelling", plot_width=1800, plot_height=1100)

    # label on x-axis
    p.xaxis.axis_label = 'X'

    # label on y-axis
    p.yaxis.axis_label = 'Y'

    # plot each datapoint as a circle
    # with custom attributes.
    p.circle(x='x',
             y='y',
             color='color',
             # fill_alpha=0.3,
             size=3,
             source=source)

    p.add_tools(hover)
    script, div = components(p)

    return script,div