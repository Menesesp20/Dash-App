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

df = pd.read_csv('C:/Users/menes/Documents/Data Hub/Database/optaMundial.csv')
df["matchTimestamp"] = 60 * df["minute"] + df["second"]
df["matchTimestamp"] = pd.to_timedelta(df["matchTimestamp"], unit='s')
df.drop_duplicates(subset=['name', 'matchTimestamp', 'team', 'typedisplayName', 'x', 'y'], keep='first', inplace=True)
df.sort_values(by=['Match_ID', 'matchTimestamp'], inplace=True, ascending=[True, True])
df.reset_index(drop=True, inplace=True)

teams = df.team.unique()

players = df.name.unique()

matchID = df.Match_ID.unique()

matchID = matchID.tolist()

matchID.insert(0, 'All Season')

style_card = style={"box-shadow": "2px 2px 10px 0px rgba(10, 9, 7, 0.10)",
                    "margin": "10px",
                    "padding": "10px",
                    "height": "100vh"}

competition = ['Mundial', 'Premier League', 'La Liga', 'Ligue 1', 'Bundesliga', 'Serie A', 'Brasileirão', 'Liga Bwin']

graphs = ['Touch Map', 'xT Map']

layout = dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label('Teams'),
                            dcc.Dropdown(
                                id='dp-teams',
                                options=teams,
                                value='Portugal')
                            ])],
                            md=4),
                    dbc.Col([
                        html.Div([
                                html.Label('Players'),
                                dcc.Dropdown(
                                        id='dp-players',
                                        options=players,
                                        value='')
                                ])],
                        md=4),

                dbc.Col([
                        html.Div([
                                html.Label('GameDay'),
                                dcc.Dropdown(
                                        id='dp-matchID',
                                        options=matchID,
                                        value='All Season')
                                ])],
                        md=2),
                ], style={"margin-top": "10px"}),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Img(id='visualization', src='', style={'width' : '70%'})
                            ]),
                        ], width={"size": 12, "offset": 2}, style={'margin-top' : '50px'}, md=10)
                    ])
                ])
            ])

@app.callback(
    Output('dp-players', 'options'),
    Input('dp-teams', 'value'),
    State('rd-competition', 'value'))

def dropdown_players(club, rdCompetition):

        if rdCompetition == 'Mundial':
                players = df.loc[(df.team == club) & (df.League == 'World Cup 2022')].reset_index(drop=True)
                
        players = players.name.unique()
        options=[{'label':name, 'value':name} for name in players]
        
        return  options

@app.callback(
    Output('dp-players', 'value'),
    Input('dp-teams', 'value'))

def dropdown_playerValue(club):

        players = df.loc[df.team == club].reset_index(drop=True)
        players = players.name.unique()
        options=[{'label':name, 'value':name} for name in players]
        
        return  options[0].get('value')


@app.callback(
    Output('visualization', 'src'),
    Input('dp-players', 'value'),
    [State('rd-competition', 'value'),
     State('rd-viz', 'value'),
     State('dp-teams', 'value'),
     State('dp-matchID', 'value')])

def visualization(Player, rdCompetition, rdViz, club, gameID):
        
        if Player == '':
                players = df.loc[df.team == club].reset_index(drop=True)
                players = players.name.unique()
                Player = players[0]
        
        if (rdCompetition == 'Mundial') & (rdViz == 'Touch Map'):
                return fc.touch_Map(club, Player, gameID)

        elif (rdCompetition == 'Mundial') & (rdViz == 'xT Map'):
                return fc.heatMap_xT(club, Player)

        elif (rdCompetition == 'Mundial') & (rdViz == 'Defensive Dashboard'):
                return fc.dashboardDeffensive(club, gameID, Player, 'WhoScored')

        elif (rdCompetition == 'Mundial') & (rdViz == 'Offensive Dashboard'):
                return fc.dashboardOffensive(club, Player, gameID, 'WhoScored')

        elif (rdCompetition == 'Mundial') & (rdViz == 'Zone 14'):
                return fc.plotDasboardZone14(club, 'WhoScored')

        elif (rdCompetition == 'Mundial') & (rdViz == 'Profile'):
                return fc.profilePlayer()