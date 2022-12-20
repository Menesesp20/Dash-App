from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib.pyplot as plt

from app import *
import Functions.Game as fc

from matplotlib import font_manager

font_path = 'C:/Users/menes/Documents/Data Hub/Fonts/Gagalin-Regular.otf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

#Courier New
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

clubColors = {'Brazil' : ['#fadb04', '#1c3474'],
              'Portugal' : ['#e1231b', '#004595'],
              'Argentina' : ['#52a9dc', '#dbe4ea'],
              'Saudi Arabia' : ['#145735', '#dbe4ea'],
              'Ghana' : ['#145735', '#dbe4ea'],
              'Serbia' : ['#FF0000', '#E8E8E8'],
              'Spain' : ['#FF0000', '#E8E8E8'],
              'Germany' : ['#aa9e56', '#FF0000'],
              'France' : ['#202960', '#d10827'],
              'Poland' : ['#d10827', '#E8E8E8'],
              'Morocco' : ['#db221b', '#044c34'],
              'Croatia' : ['#e71c23', '#3f85c5'],
              'Netherlands' : ['#f46c24', '#dcd9d7'],
              'Senegal' : ['#34964a', '#eedf36'],
              'Denmark' : ['#cb1617', '#E8E8E8'],
              'Iran' : ['#269b44', '#dd1212'],
              'Belgium' : ['#ff0000', '#e30613'],
              'USA' : ['#ff0000', '#202960'],
              'Switzerland' : ['#ff0000', '#e8e8e8'],
              'Australia' : ['#202960', '#e30613'],
              'Wales' : ['#ff0000', '#e8e8e8'],
              'Mexico' : ['#00a94f', '#ff0000'],
              'Uruguay' : ['#52a9dc', '#e8e8e8'],
              'Canada' : ['#ff0000', '#ff0000'],
              'Costa Rica' : ['#ff0000', '#202960']}

players = pd.read_csv('worldCupPlayers.csv')
stats = pd.read_csv('worldCupStats.csv')

players = players[['player_id', 'player']]

df = stats.merge(players, on='player_id')

physical = df[['player_id', 'player', 'top_speed', 'avg_speed', 'total_distance',
       'distance_high_speed_sprinting', 'distance_low_speed_sprinting',
       'distance_high_speed_running', 'distance_jogging', 'distance_walking',
       'speed_runs', 'sprints']]

players = physical.player.unique()

layout = dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                                html.Label('Players'),
                                dcc.Dropdown(
                                        id='dp-playersPhysical',
                                        options=players,
                                        value='Lionel Messi')
                                ])],
                        md=5),

                dbc.Col([
                        html.Div([
                                html.Label('Players 2'),
                                dcc.Dropdown(
                                        id='dp-playersPhysical2',
                                        options=players,
                                        value='Cristiano Ronaldo')
                                ])],
                        md=5),
                ], style={"margin-top": "10px"}),

                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Img(id='visualization-Physical', src='', style={'width' : '70%'})
                            ]),
                        ], width={"size": 12, "offset": 2}, style={'margin-top' : '50px'}, md=10)
                    ])
                ])
            ])

@app.callback(
    Output('visualization-Physical', 'src'),
    [Input('dp-playersPhysical', 'value'),
     Input('dp-playersPhysical2', 'value')],
    [State('rd-viz', 'value')])

def visualization(Player, Player2, rdViz):
    
    cols = physical.columns
    cols = list(cols)
    del cols[:2]
    
    if rdViz == 'Radar':
        return fc.radar_chartPhysical(physical, Player, cols)
            
    elif rdViz == 'Percentil':
        return fc.PizzaChartPhysical(physical, Player, cols)

    elif rdViz == 'Compare':
        return fc.pizza_ComparePhysical(physical, Player, Player2, cols)