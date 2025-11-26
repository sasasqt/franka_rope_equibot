# import plotly.graph_objects as go
# import numpy as np

# categories = ['R', 'P', 'R+P']
# # 'DDPM', '1-step inference'
# bar_names = ['Ours', 'Equibot', 'DP']
# bar_colors = ['#13343B', '#FFC185', "#A0A0A0"]  # Use brand colors, orange for 'Ours'
# # ,'#5D878F', "#EEE313"
# values = [
#     [0.241,0.405,0.253],
#     [0.244,0.009,0.0],
#     [0.331, 0.063, 0.012],
# ]

# # errors = [
# #     [0.05, 0.05, 0.05, 0.05],
# #     [0.05, 0.05, 0.05, 0.05],
# #     [0.05, 0.05, 0.05, 0.05],
# #     [0.05, 0.05, 0.05, 0.05]
# # ]

# fig = go.Figure()

# for i, (name, color) in enumerate(zip(bar_names, bar_colors)):
#     fig.add_bar(
#         x=categories,
#         y=values[i],
#         name=name,
#         # error_y=dict(type='data', array=errors[i], visible=True),
#         marker_color=color,
#         cliponaxis=False
#     )

# fig.update_layout(
#     barmode='group',
#     title_text='2 Distinct Trajectories',
#     yaxis_title='Final Reward of Trajectory Followed',
#     xaxis_title='',
#     legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
# )
# fig.update_yaxes(title_text='Final Reward of Trajectory Followed')
# fig.update_xaxes(title_text='')

# fig.write_image('grouped_bar_error.png')


# import plotly.graph_objects as go
# import numpy as np

# categories = ['R', 'P', 'R+P']
# # 'DDPM', '1-step inference'
# bar_names = ['DDIM seed 0 2 trajectories', 'DDIM seed 0 1 trajectory', 'DDIM seed 0 1 trajectory']
# bar_colors = ['#13343B', "#2B55A3", "#456650"]  # Use brand colors, orange for 'Ours'
# # ,'#5D878F', "#EEE313"
# values = [
#     [0.241,0.405,0.253],
#     [0.321,0.415,0.323],
#     [0.289,0.41,0.309],
# ]

# # errors = [
# #     [0.05, 0.05, 0.05, 0.05],
# #     [0.05, 0.05, 0.05, 0.05],
# #     [0.05, 0.05, 0.05, 0.05],
# #     [0.05, 0.05, 0.05, 0.05]
# # ]

# fig = go.Figure()

# for i, (name, color) in enumerate(zip(bar_names, bar_colors)):
#     fig.add_bar(
#         x=categories,
#         y=values[i],
#         name=name,
#         # error_y=dict(type='data', array=errors[i], visible=True),
#         marker_color=color,
#         cliponaxis=False
#     )

# fig.update_layout(
#     barmode='group',
#     title_text='Final Reward by Category',
#     yaxis_title='Final Reward',
#     xaxis_title='',
#     legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
# )
# fig.update_yaxes(title_text='Final Reward')
# fig.update_xaxes(title_text='')

# fig.write_image('grouped_bar_error.png')



# import plotly.graph_objects as go
# import numpy as np

# categories = ['R', 'P', 'R+P']
# # 'DDPM', '1-step inference'
# bar_names = ['DDIM', 'DDPM', '1-step inference']
# bar_colors = ['#13343B', "#2B55A3", "#456650"]  # Use brand colors, orange for 'Ours'
# # ,'#5D878F', "#EEE313"
# values = [
#     [0.241,0.405,0.253],
#     [0.239,0.404,0.258],
#     [0.267,0.342,0.238],
# ]

# # errors = [
# #     [0.05, 0.05, 0.05, 0.05],
# #     [0.05, 0.05, 0.05, 0.05],
# #     [0.05, 0.05, 0.05, 0.05],
# #     [0.05, 0.05, 0.05, 0.05]
# # ]

# fig = go.Figure()

# for i, (name, color) in enumerate(zip(bar_names, bar_colors)):
#     fig.add_bar(
#         x=categories,
#         y=values[i],
#         name=name,
#         # error_y=dict(type='data', array=errors[i], visible=True),
#         marker_color=color,
#         cliponaxis=False
#     )

# fig.update_layout(
#     barmode='group',
#     title_text='Final Reward by Category',
#     yaxis_title='Final Reward',
#     xaxis_title='',
#     legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
# )
# fig.update_yaxes(title_text='Final Reward')
# fig.update_xaxes(title_text='')

# fig.write_image('grouped_bar_error.png')



# import plotly.graph_objects as go
# import numpy as np

# categories = ['R', 'R+P','TR+R', 'TR+R+P']
# bar_names = ['Ours', 'Equibot', 'DP']
# bar_colors = ['#13343B', '#FFC185', "#A0A0A0"]  # Use brand colors, orange for 'Ours'
# # ,'#5D878F', "#EEE313"
# values = [
#     [0.4286,0.1705,0.7194],
#     [0.5167,0.0099,0.0099],
#     [0.2819, 0.0723, 0.1932],
#     [0.2610, 0.0099, 0.0099],
# ]

# fig = go.Figure()

# for i, (name, color) in enumerate(zip(bar_names, bar_colors)):
#     fig.add_bar(
#         x=categories,
#         y=values[i],
#         name=name,
#         # error_y=dict(type='data', array=errors[i], visible=True),
#         marker_color=color,
#         cliponaxis=False
#     )

# fig.update_layout(
#     barmode='group',
#     title_text='Pick & Place',
#     yaxis_title='3D IoU of Target',
#     xaxis_title='',
#     legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
# )
# # fig.update_yaxes(title_text='Final Reward of Trajectory Followed')
# # fig.update_xaxes(title_text='')
import plotly.graph_objects as go
import numpy as np

categories = ['R', 'R+P','TR+R', 'TR+R+P']
bar_names = ['Ours', 'Equibot', 'DP']
bar_colors = ['#13343B', '#FFC185', "#A0A0A0"]

# rows = bars (Ours, Equibot, DP), cols = categories
values = np.array([
    [0.4286, 0.5167, 0.2819, 0.2610],  # Ours
    [0.1705, 0.0099, 0.0723, 0.0099],  # Equibot
    [0.7194, 0.0099, 0.1932, 0.0099],  # DP
])

assert values.shape == (len(bar_names), len(categories))

fig = go.Figure()
for i, (name, color) in enumerate(zip(bar_names, bar_colors)):
    fig.add_bar(x=categories, y=values[i], name=name, marker_color=color, cliponaxis=False)

fig.update_layout(
    barmode='group',
    title_text='Push T',
    yaxis_title='3D IoU of Target',
    xaxis_title='',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.write_image('pusht.png')

# import plotly.graph_objects as go
# import numpy as np

# categories = ['R','TR+R', 'TR+R+P']
# bar_names = ['Ours', 'Equibot', 'DP']
# bar_colors = ['#13343B', '#FFC185', "#A0A0A0"]

# values = np.array([
#     [0.6110, 0.1227, 0.3766],
#     [0.3817, 0.0000, 0.0024],
#     [0.4083, 0.0000, 0.0024],
# ])  # shape (4,3) = (categories, bars)

# values = values.T  # shape (3,4) = (bars, categories)

# fig = go.Figure()
# for i, (name, color) in enumerate(zip(bar_names, bar_colors)):
#     fig.add_bar(x=categories, y=values[i], name=name, marker_color=color, cliponaxis=False)

# fig.update_layout(
#     barmode='group',
#     title_text='Pick & Place',
#     yaxis_title='3D IoU of Target',
#     xaxis_title='',
#     legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
# )

# fig.write_image('pnp.png')