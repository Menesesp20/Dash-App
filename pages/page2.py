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

events = ['BallRecovery', 'Pass', 'defensiveActions', 'ballLost']

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
                                html.Label('Event Type'),
                                dcc.Dropdown(
                                        id='dp-event',
                                        options=events,
                                        value='Pass')
                                ])],
                        md=2),

                dbc.Col([
                        html.Div([
                                html.Label('GameDay'),
                                dcc.Dropdown(
                                        id='dp-matchID',
                                        options=matchID,
                                        value='All Season')
                                ])],
                        md=4),

                ], style={"margin-top": "10px"}),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Img(id='visualizationTeams', src='', style={'width' : '80%'})
                            ]),
                        ], width={"size": 12, "offset": 1}, style={'margin-top' : '50px'}, md=10)
                    ])
                ])
            ])

@app.callback(
    Output('visualizationTeams', 'src'),
    Input('dp-teams', 'value'),
    [State('rd-competition', 'value'),
     State('rd-viz', 'value'),
     State('dp-matchID', 'value'),
     State('dp-event', 'value')])

def visualization(club, rdCompetition, rdViz, rdMatchID, dpEvent):
       
        if (rdCompetition == 'Mundial') & (rdViz == 'Chances Created'):
                return fc.heatMapChances(club, 'WhoScored')

        elif (rdCompetition == 'Mundial') & (rdViz == 'xT Map'):
                return fc.heatMap_xT(club)

        elif (rdCompetition == 'Mundial') & (rdViz == 'Passing Network'):
                return fc.passing_networkWhoScored(club, rdMatchID)

        elif (rdCompetition == 'Mundial') & (rdViz == 'Field Tilt'):
                return fc.field_Tilt(club, rdMatchID)

        elif (rdCompetition == 'Mundial') & (rdViz == 'High TurnOvers'):
                return fc.highTurnovers(club, rdMatchID, 'WhoScored')

        elif (rdCompetition == 'Mundial') & (rdViz == 'BuildUp'):
                return fc.draw_heatmap_construcao(club, 'WhoScored')

        elif (rdCompetition == 'Mundial') & (rdViz == 'GoalKick'):
                return fc.GoalKick(club, 'WhoScored')

        elif (rdCompetition == 'Mundial') & (rdViz == 'Corners'):
                return fc.cornersTaken(club, 'WhoScored')

        elif (rdCompetition == 'Mundial') & (rdViz == 'xT Flow'):
                return fc.xT_Flow(club, rdMatchID, 'WhoScored')
        
        elif (rdCompetition == 'Mundial') & (rdViz == 'Possession Gained'):
                return fc.possessionGained(club, dpEvent)