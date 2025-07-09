import plotly.graph_objects as go
import numpy as np

categories = ['Original', 'OOD (R+Su)', 'OOD (R+Sn)', 'OOD (R+Sn+P)']

bar_names = ['DP3', 'DP3+Aug', 'DP3 w/ Equivariance', 'Ours']
bar_colors = ['#5D878F', '#ECEBD5', '#13343B', '#FFC185']  # Use brand colors, orange for 'Ours'

values = [
    [0.9, 0.1, 0.1, 0.1],
    [0.55, 0.6, 0.55, 0.52],
    [0.75, 0.75, 0.75, 0.75],
    [0.8, 0.85, 0.85, 0.75]
]

errors = [
    [0.05, 0.05, 0.05, 0.05],
    [0.05, 0.05, 0.05, 0.05],
    [0.05, 0.05, 0.05, 0.05],
    [0.05, 0.05, 0.05, 0.05]
]

fig = go.Figure()

for i, (name, color) in enumerate(zip(bar_names, bar_colors)):
    fig.add_bar(
        x=categories,
        y=values[i],
        name=name,
        error_y=dict(type='data', array=errors[i], visible=True),
        marker_color=color,
        cliponaxis=False
    )

fig.update_layout(
    barmode='group',
    title_text='Final Reward by Category',
    yaxis_title='Final Reward',
    xaxis_title='',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)
fig.update_yaxes(title_text='Final Reward')
fig.update_xaxes(title_text='')

fig.write_image('grouped_bar_error.png')
