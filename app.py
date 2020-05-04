# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go

import json

app = dash.Dash(__name__, 
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],)
# Below is the Flask server that will deliver this application to web browsers
# of users.
server = app.server 

# Set up data files
met_data_dir = '/misc/team/_ZARNEKOW/2_data_processed/met/biomet30minute/'
met_csv_dict = {
#    2007:'ZRK_biomet_20070509T000000_20090725T180000_30min.txt',
    2013:'ZRK_biomet_20130305T003000_20140101T000000_30min.txt',
    2014:'ZRK_biomet_20140101T003000_20150101T000000_30min.txt',
    2015:'ZRK_biomet_20150101T003000_20160101T000000_30min.txt',
    2016:'ZRK_biomet_20160101T003000_20170101T000000_30min.txt',
    2017:'ZRK_biomet_20170101T003000_20180101T000000_30min.txt',
    2018:'ZRK_biomet_20180101T003000_20190101T000000_30min.txt',
    2019:'ZRK_biomet_20190101T003000_20200101T000000_30min.txt',
}
flux_data_dir = '/home/zhanli/Workspace/mefe_team/_ZARNEKOW/2_data_processed/flux'
flux_csv_dict = {
#    2007:'2007/ZRK_TGA_FluxFratini_qc.txt',
    2013:'2013/ZRK_2013_FluxFratini_qc.csv',
    2014:'2014/ZRK_2014_FluxFratini_qc.csv',
    2015:'2015/ZRK_2015_FluxFratini_qc.csv',
    2016:'2016/ZRK_2016_FluxFratini_qc.csv',
    2017:'2017/ZRK_2017_FluxFratini_qc.csv',
    2018:'2018/ZRK_2018_FluxFratini_qc.txt',
}

input_dir_dict = {
    'biomet':met_data_dir, 
    'flux':flux_data_dir,
}
input_files_dict = {
    'biomet':met_csv_dict, 
    'flux':flux_csv_dict,
}

year_options = [{'label':'{0:d}'.format(yval), 'value':yval} for yval in met_csv_dict.keys()]
candidate_colors = plotly.colors.qualitative.Set3
year_colors = {val:candidate_colors[i%len(candidate_colors)] for i, val in enumerate(met_csv_dict.keys())}

input_cols_dict = {
    incsvcat:pd.read_csv(os.path.join(input_dir_dict[incsvcat], incsvdict[2013]), 
        header=0, index_col=0, nrows=3).columns 
    for incsvcat, incsvdict in input_files_dict.items()
}
# var_options = [{'label':cval, 'value':cval} for cval in tmp_df.columns]
var_options = [
    {
        'label':incsvcat,
        'value':incsvcat, 
        'child':[{'label':cval, 'value':cval} for cval in cols]   
    } 
    for incsvcat, cols in input_cols_dict.items()
]

# Load data into memory
in_data_df_dict = {}
for k in set(list(met_csv_dict.keys())+list(flux_csv_dict.keys())):
    tmp_df_list = [None]*len(input_files_dict)
    for i, incsvcat in enumerate(input_files_dict.keys()):
        if k in input_files_dict[incsvcat].keys():
            tmp_df = pd.read_csv(os.path.join(input_dir_dict[incsvcat], 
                input_files_dict[incsvcat][k]), header=0, index_col=0)
        else:
            tmp_df = pd.DataFrame(np.zeros((2, len(input_cols_dict[incsvcat])))+np.nan, 
                index=['{0:d}-01-01 00:30'.format(k), '{0:d}-01-01 01:00'.format(k)], 
                columns=input_cols_dict[incsvcat])
        tmp_df.index = pd.to_datetime(tmp_df.index)
        tmp_df_list[i] = tmp_df.asfreq('30min')

    in_data_df_dict[k] = pd.concat(tmp_df_list, axis='columns', join='outer', 
        keys=input_files_dict.keys()).sort_index(axis=0)

# Shift the Jan 01 00:00:00 to the correct year from the previous year
for yval in range(np.min(list(in_data_df_dict.keys())), np.max(list(in_data_df_dict.keys()))+1):
    if yval in in_data_df_dict.keys():
        sflag = in_data_df_dict[yval].index.year != yval
        if yval+1 in in_data_df_dict.keys():
            in_data_df_dict[yval+1] = pd.concat([
                in_data_df_dict[yval].loc[sflag, :], 
                in_data_df_dict[yval+1]], axis=0)
        in_data_df_dict[yval] = in_data_df_dict[yval].loc[np.logical_not(sflag), :]

year_dummy=2012
def years2DummyYear(dt):
    return pd.DateOffset(years=year_dummy-dt.year)
single_year_ts_index = pd.date_range('{0:d}-01-01 00:00:00'.format(year_dummy), \
        '{0:d}-01-01 00:00:00'.format(year_dummy+1), freq='30min')
sflag = single_year_ts_index.year == year_dummy
single_year_ts_index = single_year_ts_index[sflag]

def genFigTimeSeries(df_dict):
    df = next(iter(df_dict.values()))
    nvars = df.shape[1]
    var_names = df.columns
    plotly_fig = plotly.subplots.make_subplots(rows=nvars, cols=1)

    for i in range(nvars):
        for k, df in df_dict.items():
            plotly_fig.add_trace(
                go.Scattergl(
                    name='{0:d}, {1:s}'.format(k, df.columns[i]), 
                    x=df.index, 
                    y=df.iloc[:, i], 
                    mode='markers', 
                    marker={
                        'size':3, 
                        'color':year_colors[k], 
                    },
                ), 
                row=i+1, 
                col=1,
            )

    plotly_fig.update_layout(showlegend=True)
    plotly_fig.update_xaxes(dict(matches='x'))
    for i in range(nvars):
        plotly_fig.update_yaxes(title_text=var_names[i], row=i+1, col=1)
    plotly_fig.update_layout(width=None, height=None)

    return plotly_fig

def genFigStackTimeSeries(df_dict):
    df = next(iter(df_dict.values()))
    nvars = df.shape[1]
    var_names = df.columns
    plotly_fig = plotly.subplots.make_subplots(rows=nvars, cols=1)

    for i, vnval in enumerate(var_names):
        for k, df in df_dict.items():
            plotly_fig.add_trace(
                go.Scattergl(
                    name='{0:d}, {1:s}'.format(k, str(vnval)), 
                    x=[val+years2DummyYear(df.index[0]) for val in df.index], 
                    y=df.iloc[:, i], 
                    mode='markers', 
                    marker={
                        'size':3, 
                        'color':year_colors[k], 
                    },
                    hovertemplate='%{{x|%b %d}}, {0:d}, %{{x|%H:%M}}, %{{y}}'.format(k), 
                ), 
                row=i+1, 
                col=1,
            )

    plotly_fig.update_layout(showlegend=True)
    plotly_fig.update_xaxes(dict(matches='x'))
    for i in range(nvars):
        plotly_fig.update_yaxes(title_text=str(var_names[i]), row=i+1, col=1)
    plotly_fig.update_xaxes(
        tickformat='%H:%M<br>%b %d', 
    )
    plotly_fig.update_layout(width=None, height=None)

    return plotly_fig

def genFigScatter(df, selectedpoints=None):
    plotly_fig = go.Figure()

    plotly_fig.add_trace(
        go.Scattergl(
            name='{0[0]:s} vs {0[1]:s}'.format(df.columns), 
            showlegend=False, 
            x=df.iloc[:, 0], 
            y=df.iloc[:, 1], 
            mode='markers',
            marker={
                'size':3, 
                'color':year_colors[df.index[0].year], 
            },
            selectedpoints=selectedpoints,
            unselected={
                'marker': { 'opacity': 0.00 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }, 
            }, 
        ),
    )

    plotly_fig.update_xaxes(title_text=df.columns[0])
    plotly_fig.update_yaxes(title_text=df.columns[1])
    plotly_fig.update_layout(width=None, height=None)

    return plotly_fig

def genFigMultiScatter(df_dict, selectedpoints_dict):
    plotly_fig = go.Figure()

    for k, df in df_dict.items():
        plotly_fig.add_trace(
            go.Scattergl(
                name='{0:d}<br>{1:s} vs {2:s}'.format(k, 
                    str(df.columns[0]), 
                    str(df.columns[1])), 
                showlegend=True, 
                x=df.iloc[:, 0], 
                y=df.iloc[:, 1], 
                mode='markers',
                marker={
                    'size':3,
                    'color':year_colors[k], 
                },
                selectedpoints=selectedpoints_dict[k],
                unselected={
                    'marker': { 
                        'opacity': 0.00, 
                    },
                }, 
            ),
        )

    plotly_fig.update_layout(showlegend=True)
    plotly_fig.update_xaxes(title_text=str(df.columns[0]))
    plotly_fig.update_yaxes(title_text=str(df.columns[1]))
    plotly_fig.update_layout(width=None, height=None)

    return plotly_fig

app.layout = html.Div(
    children=[
        html.H1(
            children='Zarnekow Site', 
            style={
                'text-align':'center',
            }
        ),
        html.Div(
            children='''
                Processed data at 30-min interval.
            ''', 
            style={
                'text-align':'center', 
            }, 
        ),
        html.Div(
            id='control-card', 
            children=[
                html.P(
                        'Select Year', 
                        style={
                            'font-weight':'bold', 
                        },
                    ),
                dcc.Dropdown(
                    id='select-year', 
                    options=year_options, 
                    value=[year_options[0]['value'],],
                    multi=True, 
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.P(
                                    'Select Variable #1', 
                                    style={
                                        'font-weight':'bold',
                                    }, 
                                ),
                                html.P(
                                    'Category', 
                                    style={
                                        'font-weight':'normal',
                                    }, 
                                ),
                                dcc.Dropdown(
                                    id='select-var1-1', 
                                    options=[{k:val[k] for k in ['label', 'value']} for val in var_options], 
                                    value=var_options[0]['value'], 
                                    clearable=False, 
                                ), 
                                html.P(
                                    id='title-var1-2', 
                                    children=var_options[0]['label'], 
                                    style={
                                        'font-weight':'normal',
                                    }, 
                                ),
                                dcc.Dropdown(
                                    id='select-var1-2',
                                    options=var_options[0]['child'], 
                                    value=var_options[0]['child'][0]['value'],
                                    clearable=False, 
                                ), 
                            ], 
                            style={
                                'float':'left',
                                'width':'50%',
                                'padding':'0 10px 10px 0', 
                            }, 
                        ),
                        html.Div(
                            children=[
                                html.P(
                                    'Select Variable #2', 
                                    style={
                                        'font-weight':'bold',
                                    },
                                ),
                                html.P(
                                    'Category', 
                                    style={
                                        'font-weight':'normal',
                                    }, 
                                ),
                                dcc.Dropdown(
                                    id='select-var2-1',
                                    options=[{k:val[k] for k in ['label', 'value']} for val in var_options], 
                                    value=var_options[0]['value'],
                                    clearable=False, 
                                ),
                                html.P(
                                    id='title-var2-2', 
                                    children=var_options[0]['label'], 
                                    style={
                                        'font-weight':'normal',
                                    }, 
                                ),
                                dcc.Dropdown(
                                    id='select-var2-2',
                                    options=var_options[0]['child'], 
                                    value=var_options[0]['child'][1]['value'],
                                    clearable=False, 
                                ), 
                            ], 
                            style={
                                'float':'left',
                                'width':'50%', 
                                'padding':'0 0 10px 10px', 
                            }, 
                        ), 
                    ],
                    style={
                        'display':'flex', 
                    }, 
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.P(
                                    'Time Series Plots', 
                                    style={
                                        'text-align':'center', 
                                    }, 
                                ), 
                                dcc.Graph(
                                    id='fig-stack-time-series', 
                                    style={
                                        'height':'600px',
                                    },
                                ), 
                            ],
                            style={
                                'float':'left',
                                'width':'60%',
                            }, 
                        ), 
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.P(
                                            'Scatter Plot', 
                                            style={
                                                'text-align':'right', 
                                                'float':'left',
                                                'width':'20%',
                                                'padding':'0 10px 0 0',
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id='select-year-scatter', 
                                            multi=True, 
                                            style={
                                                'float':'left',
                                                'width':'80%',
                                            },
                                            clearable=True, 
                                        ),
                                    ],
                                    style={
                                        'display':'flex',
                                        'align-items':'center', 
                                    }, 
                                ), 
                                dcc.Graph(
                                    id='fig-scatter',
                                    style={
                                        'height':'600px',
                                    },
                                ), 
                            ], 
                            style={
                                'float':'left',
                                'width':'40%', 
                            }, 
                        ), 
                    ], 
                    style={
                        'display':'flex', 
                    }, 
                ), 
            ], 
        ),
        # Hidden div inside the app that stores the intermediate value
        html.Div(
            id='ts-info', 
            style={'display': 'none'},
        ),
        html.Div(
            id='ts-bounds', 
            style={'display': 'none'},
        ),
    ],
)

@app.callback(
    [
        Output('select-year-scatter', 'options'), 
        Output('select-year-scatter', 'value'), 
    ], 
    [
        Input('select-year', 'value'), 
    ],
    [
        State('select-year-scatter', 'options'), 
        State('select-year-scatter', 'value'),
    ],
)
def setYears4Scatter(
    select_years, 
    cur_scatter_year_options, 
    cur_scatter_year_value,
):
    if len(select_years) == 0:
        return cur_scatter_year_options, cur_scatter_year_value
    else:
        return [val for val in year_options if val['value'] in select_years], \
               [select_years[0],], 

@app.callback(
    [
        Output('select-var1-2', 'options'), 
        Output('select-var1-2', 'value'), 
        Output('title-var1-2', 'children'), 
    ], 
    [
        Input('select-var1-1', 'value'), 
    ], 
)
def setVar1_2(
    var1_1, 
):
    tmpidx = 0
    for i in range(len(var_options)):
        if var_options[i]['value'] == var1_1:
            tmpidx = i
            break
    options=var_options[tmpidx]['child']
    value=var_options[tmpidx]['child'][0]['value']
    return options, value, var_options[tmpidx]['label']

@app.callback(
    [
        Output('select-var2-2', 'options'), 
        Output('select-var2-2', 'value'), 
        Output('title-var2-2', 'children'),
    ], 
    [
        Input('select-var2-1', 'value'), 
    ], 
)
def setVar1_2(
    var2_1, 
):
    tmpidx = 0
    for i in range(len(var_options)):
        if var_options[i]['value'] == var2_1:
            tmpidx = i
            break
    options=var_options[tmpidx]['child']
    value=var_options[tmpidx]['child'][0]['value']
    return options, value, var_options[tmpidx]['label']

@app.callback(
    [
        Output('ts-info', 'children'),
    ], 
    [
        Input('select-year', 'value'), 
        Input('select-var1-2', 'value'),
        Input('select-var2-2', 'value'), 
    ],
    [
        State('ts-info', 'children'), 
        State('select-var1-1', 'value'), 
        State('select-var2-1', 'value'), 
    ], 
)
def updateTimeSeriesInfo(
    year, 
    var1, var2, 
    ts_info_json, var1_1, var2_1, 
):
    if len(year)==0 or var1 is None or var2 is None:
        return ts_info_json, 
    else:
        ts_info = {
            'year':year, 
            'var1':(var1_1, var1), 
            'var2':(var2_1, var2), 
        }
        return json.dumps(ts_info), 

@app.callback(
    [
        Output('fig-stack-time-series', 'figure'), 
    ], 
    [
        Input('ts-info', 'children'), 
    ],
)
def updateFigTimeSeries(
    ts_info_json
):
    ts_info = json.loads(ts_info_json)
    year_list, var1, var2 = ts_info['year'], ts_info['var1'], ts_info['var2']
    var1, var2 = tuple(var1), tuple(var2)
    return genFigStackTimeSeries({val:in_data_df_dict[val][[var1, var2]] for val in year_list}),

@app.callback(
    [
        Output('ts-bounds', 'children'), 
    ], 
    [
        Input('ts-info', 'children'), 
        Input('fig-stack-time-series', 'relayoutData'),
        Input('select-year-scatter', 'value'), 
    ],
    [
        State('fig-stack-time-series', 'figure'), 
        State('ts-bounds', 'children'), 
    ], 
)
def updateTimeSeriesBounds(
    ts_info_json, relayoutData, scatter_year_list, 
    fig_ts, cur_ts_bounds_json, 
):
    ctx = dash.callback_context
    if len(scatter_year_list) == 0:
        return cur_ts_bounds_json, 

    ts_info = json.loads(ts_info_json)
    year_list,var1, var2 = ts_info['year'], ts_info['var1'], ts_info['var2']
    var1, var2 = tuple(var1), tuple(var2)
    df_list = [in_data_df_dict[val][[var1, var2]] for val in year_list]
    nvars = df_list[0].shape[1]

    # Default bounds are data min and max. 
    ts_bounds = {
        'xbounds':[[
            str(np.min([df.index[0]+years2DummyYear(df.index[0]) for df in df_list])), \
            str(np.max([df.index[-1]+years2DummyYear(df.index[0]) for df in df_list]))] 
            for i in range(nvars)], 
        'ybounds':[[
            np.min([df.iloc[:, i].min() for df in df_list]), 
            np.max([df.iloc[:, i].max() for df in df_list])] 
            for i in range(nvars)],
        'years':scatter_year_list, 
    }
    if  'ts-info.children' in [val['prop_id'] for val in ctx.triggered]:
        return json.dumps(ts_bounds),

    # Update the bounds from the current axes range in the figure of time series.
    if fig_ts is not None:
        for i in range(nvars):
            tmpstr = 'xaxis' if i==0 else 'xaxis{0:d}'.format(i+1)
            if fig_ts['layout'][tmpstr]['range'] is not None:
                tmp = [ pd.to_datetime(val) for val in fig_ts['layout'][tmpstr]['range'] ]
                ts_bounds['xbounds'][i] = [
                    str(single_year_ts_index[0] if tmp[0]<single_year_ts_index[0] else tmp[0]), 
                    str(single_year_ts_index[-1] if tmp[1]>single_year_ts_index[-1] else tmp[1]), 
                ]
            tmpstr = 'yaxis' if i==0 else 'yaxis{0:d}'.format(i+1)
            if fig_ts['layout'][tmpstr]['range'] is not None:
                ts_bounds['ybounds'][i] = fig_ts['layout'][tmpstr]['range']

    return json.dumps(ts_bounds), 

@app.callback(
    Output('fig-scatter', 'figure'), 
    [
        Input('ts-info', 'children'), 
        Input('ts-bounds', 'children'),
    ],
)
def updateFigScatter(
    ts_info_json, ts_bounds_json, 
):
    ts_info = json.loads(ts_info_json)
    year,var1, var2 = ts_info['year'], ts_info['var1'], ts_info['var2']
    var1, var2 = tuple(var1), tuple(var2)

    ts_bounds = json.loads(ts_bounds_json)
    scatter_year_list = ts_bounds['years']

    df_dict = {}
    selectedpoints_dict = {}
    for scatter_year in scatter_year_list:
        df = in_data_df_dict[scatter_year][[var1, var2]]
        df_dict[scatter_year] = df

        sflag = np.ones_like(df.index, dtype=np.bool_)
        for rgval in ts_bounds['xbounds']:
            tmp_yeardiff = pd.DateOffset(years=scatter_year-pd.to_datetime(rgval[0]).year)
            sflag = np.logical_and(
                        sflag, 
                        np.logical_and(
                            df.index>=pd.to_datetime(rgval[0])+tmp_yeardiff, 
                            df.index<=pd.to_datetime(rgval[1])+tmp_yeardiff
                        )
                    )
        for i, rgval in enumerate(ts_bounds['ybounds']):
            sflag = np.logical_and(
                        sflag, 
                        np.logical_and(df.iloc[:, i]>=rgval[0], df.iloc[:, i]<=rgval[1])
                    )
        if np.sum(sflag) < df.shape[0]:
            selectedpoints_dict[scatter_year], = np.nonzero(sflag.values)
        else:
            selectedpoints_dict[scatter_year] = None

    return genFigMultiScatter(df_dict, selectedpoints_dict)

if __name__ == '__main__':
    app.run_server(debug=True, port=8901)
