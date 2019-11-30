# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:24:05 2018

@author: michaelek
"""
try:
    import plotly.offline as py
    import plotly.graph_objs as go
except:
    print('install plotly for plot functions to work')


def plot(self, input_site, output_path='nat_flow.html', title='Naturalisation', yaxis_label='Flow (m3/s)', line_width=2, axis_font_size=18, hover_font_size=18, legend_font_size=18, plot_bg=None, showgrid=False):
    """
    Function to run and plot the detide results.

    Parameters
    ----------
    output_path : str
        Path to save the html file.

    Returns
    -------
    DataFrame or Series
    """

    if hasattr(self, 'nat_flow'):
        nat_flow = self.nat_flow.copy()
    else:
        nat_flow = self.naturalisation()

    nat_flow1 = nat_flow.loc[:, (slice(None), input_site)]
    nat_flow1.columns = nat_flow1.columns.droplevel(1)

    colors1 = ['rgb(102,194,165)', 'rgb(252,141,98)', 'rgb(141,160,203)']

    fig = go.Figure(layout=dict(
        title=title,
        yaxis={'title': yaxis_label},
        dragmode='pan',
        xaxis_rangeslider_visible=False,
        legend=dict(font=dict(size=legend_font_size)),
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg))

    fig.add_scattergl(
        x=nat_flow1.index,
        y=nat_flow1['Flow'],
        name = 'Recorded Flow',
        line = dict(color = colors1[0]),
        opacity = 0.8,
        line_width=line_width,
        hoverlabel=dict(font=dict(size=hover_font_size)))

    fig.add_scattergl(
        x=nat_flow1.index,
        y=nat_flow1['SwUsageRate'],
        name = 'Stream Usage',
        line = dict(color = colors1[1]),
        opacity = 0.8,
        line_width=line_width,
        hoverlabel=dict(font=dict(size=hover_font_size)))

    fig.add_scattergl(
        x=nat_flow1.index,
        y=nat_flow1['NatFlow'],
        name = 'Naturalised Flow',
        line = dict(color = colors1[2]),
        opacity = 0.8,
        line_width=line_width,
        hoverlabel=dict(font=dict(size=hover_font_size)))

    config = {"displaylogo": False, 'scrollZoom': True, 'showLink': False}

#    fig = dict(data=data, layout=layout)
    fig.update_xaxes(title_font=dict(size=axis_font_size), tickfont=dict(size=axis_font_size), showgrid=showgrid)
    fig.update_yaxes(title_font=dict(size=axis_font_size), tickfont=dict(size=axis_font_size), showgrid=showgrid)

    py.plot(fig, filename = output_path, config=config)

    return nat_flow1
