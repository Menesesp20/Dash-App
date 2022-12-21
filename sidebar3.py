from dash import html, dcc
import dash_bootstrap_components as dbc
from app import app


style_card = style={"box-shadow": "2px 2px 10px 0px rgba(10, 9, 7, 0.10)",
                    "margin": "10px",
                    "padding": "10px",
                    "height": "95vh"}

tiers = [0, 1, 2, 3, 4]

graphs = ['Scouting List', 'Styles of Play', 'Player Report', 'Player Stats', 'Player Radar']

roles = ['Box Forward', 'False 9', 'Advanced Forward', 'Target Man']

# =========  Layout  =========== #
layout = dbc.Card(
    [
        html.H2("DATA HUB", style={'font-size': '30px', 'color': '#e8e8e8'}),
        html.Hr(), 
        html.P("Data driven approach", style={'color': '#e8e8e8'}, className="lead"),
        html.H5("Tiers:", style={"margin-top": "20px", 'color': '#e8e8e8'}),
        dcc.RadioItems(tiers,
                       tiers[0], id="rd-tiers",
                        inputStyle={"margin-right": "30px", "margin-left": "5px", 'color': '#e8e8e8'}),

        html.H5("Roles:", style={"margin-top": "20px", 'color': '#e8e8e8'}),
        dcc.RadioItems(roles,
                       roles[0], id="rd-roles",
                        inputStyle={"margin-right": "30px", "margin-left": "5px", 'color': '#e8e8e8'}),
        
        html.H5("Visualizations:", style={"margin-top": "20px", 'color': '#e8e8e8'}),
        dcc.RadioItems(graphs,
                       graphs[0], id="rd-viz",
                        inputStyle={"margin-right": "30px", "margin-left": "5px", 'color': '#e8e8e8'}),
        
        html.H5("Pages:", style={"margin-top": "20px", 'color': '#e8e8e8'}),
        dbc.Nav([
            dbc.NavLink("Players", href="/", active="exact"),
            dbc.NavLink("Teams", href="/page2", active="exact"),
            dbc.NavLink("Scouting", href="/page3", active="exact"),
            dbc.NavLink("Physical", href="/page4", active="exact"),
            ])
        
        ],style=style_card)