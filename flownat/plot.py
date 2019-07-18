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


def plot(self, input_site, output_path='nat_flow.html', title='Naturalisation', yaxis_label='water level (m)'):
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

    orig = go.Scattergl(
        x=nat_flow1.index,
        y=nat_flow1['Flow'],
        name = 'Recorded Flow',
        line = dict(color = colors1[0]),
        opacity = 0.8)

    usage = go.Scattergl(
        x=nat_flow1.index,
        y=nat_flow1['SwUsageRate'],
        name = 'Stream Usage',
        line = dict(color = colors1[1]),
        opacity = 0.8)

    nat = go.Scattergl(
        x=nat_flow1.index,
        y=nat_flow1['NatFlow'],
        name = 'Naturalised Flow',
        line = dict(color = colors1[2]),
        opacity = 0.8)

    data = [orig, usage, nat]

    layout = dict(
        title=title,
        yaxis={'title': yaxis_label},
        dragmode='pan',
        xaxis_rangeslider_visible=True)

    config = {"displaylogo": False, 'scrollZoom': True, 'showLink': False}

    fig = dict(data=data, layout=layout)
    py.plot(fig, filename = output_path, config=config)

    return nat_flow1
