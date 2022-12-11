from dash import html, dcc
import dash_bootstrap_components as dbc
from app import app


style_card = style={"box-shadow": "2px 2px 10px 0px rgba(10, 9, 7, 0.10)",
                    "margin": "10px",
                    "padding": "10px",
                    "height": "95vh"}

competition = ['Mundial', 'Premier League', 'La Liga', 'Ligue 1', 'Bundesliga', 'Serie A']

graphs = ['Touch Map', 'xT Map', 'Defensive Dashboard', 'Offensive Dashboard', 'Zone 14', 'Profile']

# =========  Layout  =========== #
layout = dbc.Card(
    [
        html.H2("DATA HUB", style={'font-size': '30px', 'color': '#e8e8e8'}),
        html.Hr(), 
        html.P("Data driven approach", style={'color': '#e8e8e8'}, className="lead"),
        html.H5("Competitions:", style={"margin-top": "20px", 'color': '#e8e8e8'}),
        dcc.RadioItems(competition,
                       "Mundial", id="rd-competition",
                        inputStyle={"margin-right": "30px", "margin-left": "5px", 'color': '#e8e8e8'}),
        
        html.H5("Visualizations:", style={"margin-top": "20px", 'color': '#e8e8e8'}),
        dcc.RadioItems(graphs,
                       graphs[0], id="rd-viz",
                        inputStyle={"margin-right": "30px", "margin-left": "5px", 'color': '#e8e8e8'}),
        
        html.H5("Pages:", style={"margin-top": "20px", 'color': '#e8e8e8'}),
        dbc.Nav([
            dbc.NavLink("Players", href="/", active="exact"),
            dbc.NavLink("Teams", href="/page2", active="exact"),
            ])
        
        ],style=style_card)