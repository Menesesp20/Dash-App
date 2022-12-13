from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib.pyplot as plt

from app import *
import Functions.Game as fc
from pages import page1, page2
import sidebar1
import sidebar2

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

app.layout = dbc.Container([
                dbc.Row([
                        dbc.Col([
                            dcc.Location(id="url"), 
                            html.Div(id="sidebar-content")
                        ], md=2),

                        dbc.Col([
                            html.Div(id="page-content")
                        ]),
                    ])
            ], style={"padding": "0px"}, fluid=True)

@app.callback(
        Output("sidebar-content", "children"),
        [Input("url", "pathname")])

def render_sidebar_content(pathname):
    if pathname == "/":
        return sidebar1.layout

    if pathname == "/page2":
        return sidebar2.layout

@app.callback(
        Output("page-content", "children"),
        [Input("url", "pathname")])

def render_page_content(pathname):
    if pathname == "/":
        return page1.layout

    if pathname == "/page2":
        return page2.layout

# RUN APP
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload = False)