from bokeh.resources import INLINE
from bokeh.embed import components
from main import *
from datetime import date
from flask import Flask, render_template
from topicDataframe import *
from waitress import serve
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.io import show
from CircleBokeh import circleBokeh
from TableBokeh import tableBokeh
from main import data_df

app = Flask(__name__)
today = str(date.today())

@app.route('/model')
def index():
    result = main(data_df)[0]
    data = main(data_df)[1]

    script,div = circleBokeh(result,data)

    return render_template(
        'index.html',
        plot_script=script,
        plot_div=div,
        js_resources=INLINE.render_js(),
        css_resources=INLINE.render_css(),
    ).encode(encoding='UTF-8')

@app.route('/model//topics')
def topics():
    result = main(data_df)[0]
    data = main(data_df)[1]

    topic_df = topicDataFrame(data, result)

    script,div = tableBokeh(topic_df)

    return render_template(
        'topicIndex.html',
        plot_script=script,
        plot_div=div,
        js_resources=INLINE.render_js(),
        css_resources=INLINE.render_css(),
    ).encode(encoding='UTF-8')


if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=8080,threads=6)
