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

df = pd.read_csv('WyScout.csv')

leagues = df.Comp.unique()

players = df.Player.unique()

position = df.Position.unique()

events = ['Scouting List', 'Styles of Play', 'Player Report', 'Player Stats']

layout = dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label('Leagues'),
                            dcc.Dropdown(
                                id='dp-leagues',
                                options=leagues,
                                value='Premier League')
                            ])],
                            md=4),
                    dbc.Col([
                        html.Div([
                                html.Label('Players'),
                                dcc.Dropdown(
                                        id='dp-playersWyScout',
                                        options=players,
                                        value='')
                                ])],
                        md=4),

                dbc.Col([
                        html.Div([
                                html.Label('Position'),
                                dcc.Dropdown(
                                        id='dp-position',
                                        options=position,
                                        value='CF')
                                ])],
                        md=2),
                ], style={"margin-top": "10px"}),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Img(id='visualization-Scouting', src='', style={'width' : '70%'})
                            ]),
                        ], width={"size": 12, "offset": 2}, style={'margin-top' : '50px'}, md=10)
                    ])
                ])
            ])

@app.callback(
    Output('dp-playersWyScout', 'options'),
    Input('dp-leagues', 'value'))

def dropdown_players(league):

        players = df.loc[df.Comp == league].reset_index(drop=True)
                
        players = players.Player.unique()
        options=[{'label':name, 'value':name} for name in players]
        
        return  options

@app.callback(
    Output('dp-playersWyScout', 'value'),
    Input('dp-leagues', 'value'))

def dropdown_playerValue(league):

        players = df.loc[df.Comp == league].reset_index(drop=True)
        players = players.Player.unique()
        options=[{'label': player, 'value': player} for player in players]
        
        return  options[0].get('value')


@app.callback(
    Output('visualization-Scouting', 'src'),
    Input('dp-playersWyScout', 'value'),
    [State('rd-viz', 'value'),
     State('rd-tiers', 'value'),
     State('dp-leagues', 'value'),
     State('dp-position', 'value'),
     State('rd-roles', 'value')])

def visualization(Player, rdViz, rdTiers, league, pos, role):
        
        teamValue = df.loc[(df.Player == Player) & (df.Season == '2021/22')].reset_index(drop=True)
        teamValue = teamValue.Team.unique()
        teamValue = teamValue[0]
        
        if Player == '':
                players = df.loc[df.Comp == league].reset_index(drop=True)
                players = players.Player.unique()
                Player = players[0]
        
        if rdViz == 'Player Report':
                return fc.scoutReport(Player, teamValue, False, False, rdTiers)

        elif rdViz == 'Styles of Play':
                dfTable = fc.dataFrameForward('No', rdTiers, role, league)
                return fc.table(dfTable, role)        
        