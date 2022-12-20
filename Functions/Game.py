import pandas as pd
import numpy as np
import json
import sys
import random

import ast

import pymysql

from datetime import date

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba
from matplotlib import cm
from matplotlib import colorbar
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from matplotlib.patches import ArrowStyle
from matplotlib.patches import Circle

from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from mplsoccer import VerticalPitch, Pitch, Radar, FontManager, grid, PyPizza

import scipy.stats as stats

from highlight_text import  ax_text, fig_text

from soccerplots.utils import add_image

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

import abs_import

import statistics
import math
import ipywidgets as widgets

from IPython.display import display, Math, Latex

import warnings

plt.rcParams["figure.dpi"] = 300

from matplotlib import font_manager

from app import *

sys.path.append('Functions/visualization')

from Functions.utils import read_json
from Functions.visualization.passing_network import draw_pitch, draw_pass_map

from plottable import ColumnDefinition, ColumnType, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import circled_image, image, progress_donut

clubColors = {'Brazil' : ['#fadb04', '#1c3474'],
              'Portugal' : ['#e1231b', '#004595'],
              'Argentina' : ['#52a9dc', '#dbe4ea'],
              'Saudi Arabia' : ['#145735', '#dbe4ea'],
              'Ghana' : ['#145735', '#dbe4ea'],
              'Serbia' : ['#FF0000', '#ffffff'],
              'Spain' : ['#FF0000', '#ffffff'],
              'Germany' : ['#aa9e56', '#FF0000'],
              'France' : ['#202960', '#d10827'],
              'Poland' : ['#d10827', '#ffffff'],
              'Morocco' : ['#db221b', '#044c34'],
              'Croatia' : ['#e71c23', '#3f85c5'],
              'Netherlands' : ['#f46c24', '#dcd9d7'],
              'Senegal' : ['#34964a', '#eedf36'],
              'Denmark' : ['#cb1617', '#ffffff'],
              'Iran' : ['#269b44', '#dd1212'],
              'Belgium' : ['#ff0000', '#e30613'],
              'USA' : ['#ff0000', '#202960'],
              'Switzerland' : ['#ff0000', '#ffffff'],
              'Australia' : ['#202960', '#e30613'],
              'Wales' : ['#ff0000', '#ffffff'],
              'Mexico' : ['#00a94f', '#ff0000'],
              'Uruguay' : ['#52a9dc', '#ffffff'],
              'Canada' : ['#ff0000', '#ff0000'],
              'Costa Rica' : ['#ff0000', '#202960'],
              'Catar' : ['#7f1244', '#ffffff'],
              'Ecuador' : ['#ffce00', '#002255'],
              'South Korea' : ['#021858', '#ffffff']}

df = pd.read_csv('C:/Users/menes/Documents/Data Hub/Database/optaMundial.csv')
df["matchTimestamp"] = 60 * df["minute"] + df["second"]
df["matchTimestamp"] = pd.to_timedelta(df["matchTimestamp"], unit='s')
df.drop_duplicates(subset=['name', 'matchTimestamp', 'team', 'typedisplayName', 'x', 'y'], keep='first', inplace=True)
df.sort_values(by=['Match_ID', 'matchTimestamp'], inplace=True, ascending=[True, True])
df.reset_index(drop=True, inplace=True)

tier0 = ['La Liga', 'Premier League', 'Serie A', 'Ligue 1', 'Bundesliga']

tier1 = ['Liga Bwin', 'Brasileirao', 'Eredivisie', 'Super Liga Turkey', 'Liga Mexico', 'First Divison Belgium', 'First Division Belgium',
'Championship', 'Liga Profesional Argentina']

tier2= ['MLS', 'Austrian Bundesliga', 'NHL', 'Super League Greece', 'Swiss Super League', 'Serie B',
'Ligue 2', 'Super Liga Denamark', 'J1 League', 'K League 1', 'Smart Bank', 'J1 League', 'Ukraine League',
'Russia League', 'Scottish Premiership', 'HNL', 'Bundesliga 2', 'Serie B Italy', 'Arabia Saudi Pro League', 'UAE Pro League']

tier3 = ['1 RFEF',
'Allsvenskan',
'Liga 3 Portugal',
'Brasileirao B',
'Fortuna Liga',
'Campeonato Peruano',
'Super Liga Serbia',
'First Division Andorra',
'Superlig Albania',
'Liga SABSEG',
'Belgium Divison B',
'Liga BetPlay Colombia',
'Venezuela League', 'Ecuador Liga Pro',
'Hungarian SuperLiga', 'Romania SuperLiga',
'Poland League', 'Denamark 1st Division', 'Liga Profesional Argentina 2',
'Cyprus League', 'Uruguay Primera División', 'Chile Primera División', 'Bosnia League', 'Slovenia League',
'Moldavia League',
'Slovakia League', 'India Super League', 'Finland League', 'Lituania League', 'Scotland Championship']

tier4 = ['2 RFEF', 'Canadian Premier League', 'MLS PRO', 'USL Championship', 'USL League 1', 'Brasileirao C',
'Turkey League 2',
'Bolivia League',
'Bulgaria League',
'Israel League',
'Letonia League',
'Uruguai Segunda Divisón',
'Bundesliga 3',
'Armenia Premier League',
'Scotland League One',
'Denmark Division 3',
'Persha Liga Ukraine',
'Brazil Serie C',
'Romania Division 2',
'Hungarian Division 2',
'Pvra Liga',
'India Divison 2',
'Israel Divison 2',
'England League Two',
'FNL',
'Norway Division 2',
'Poland Fortuna 1 Liga',
'Peru Division 2',
'England League One',
'Italy Serie C',
'Italy Serie D',
'Italy Seriea D',
'Guatemala League',
'Montenegro League',
'Estonia League',
'Malta League',
'North Ireland Premiership',
'Azerbeijan League',
'Costa Rica Primera Divisón',
'Chile Primera B',
'Austria 2',
'France National 1',
'Swiss 2',
'Eredivisie 2',
'K2 League',
'J2 League',
'Luxembourg League',
'New Zealand League',
'Singapore League',
'Panama League',
'kyrgyzstan Premier League',
'Hong Kong League',
'El Salvador Primera Divisón',
'Vietnam League',
'North Macedonia First League',
'Malaysia Super Liga',
'Indonesia Liga 1',
'Georgia League',
'Belarus Premier League',
'Thai League',
'Uzbekistan League',
'Kazaskitao League',
'Algeria Ligue 1',
'A-League',
'Iran Super League',
'China Super League',
'South Africa League',
'Regionalliga Bayern',
'Regionalliga Nordost',
'Regionalliga Nord',
'Regionalliga West',
'Regionalliga SudWest',
'Finland Division 2']

center_Back = ['Non-penalty goals/90', 'Offensive duels %', 'Progressive runs/90',
                'Passes %', 'Forward passes %', 'Forward passes/90', 'Progressive passes/90',
               'PAdj Interceptions', 'PAdj Sliding tackles', 'Defensive duels/90', 'Defensive duels %',
               'Aerial duels/90', 'Aerial duels %', 'Shots blocked/90']

center_Back_bs = ['Progressive runs/90', 'Forward passes %', 'Progressive passes/90',
               'PAdj Interceptions', 'PAdj Sliding tackles', 'Aerial duels %']

full_Back = ['Successful dribbles %', 'Touches in box/90', 'Offensive duels %', 'Progressive runs/90', 'Crosses/90', 'Deep completed crosses/90',
            'Passes %', 'Deep completions/90', 'Progressive passes/90', 'Key passes/90', 'Third assists/90',
             'PAdj Interceptions', 'Defensive duels %', 'Aerial duels/90', 'Aerial duels %']

defensive_Midfield  = ['xG/90', 'Shots/90', 'Progressive runs/90', 'Successful dribbles %',
                       'Passes %', 'Forward passes %', 'Forward passes/90', 'Progressive passes/90','PAdj Sliding tackles',
                       'PAdj Interceptions', 'Aerial duels %', 'Defensive duels %', 'Offensive duels %']

Midfield  = ['xG/90', 'Shots', 'Progressive runs/90', 'Successful dribbles %',
             'Passes %', 'Forward passes %', 'Forward passes/90', 'Progressive passes/90',
             'Key passes/90', 'Second assists/90', 'Assists', 'xA',
             'PAdj Interceptions', 'Aerial duels %', 'Defensive duels %']

offensive_Midfield = ['xG/90', 'Goals/90', 'Progressive runs/90', 'Successful dribbles %',
                      'xA/90', 'Deep completions/90', 'Passes to penalty area/90',
                      'Touches in box/90', 'Key passes/90', 'Passes final 1/3 %',
                      'Passes penalty area %', 'Progressive passes/90',
                      'Succ defensive actions/90', 'PAdj Interceptions', 'Aerial duels %', 'Defensive duels %']

offensive_Midfield_BS = ['Successful dribbles %', 'xA/90', 'Deep completions/90', 'Passes to penalty area/90',
                      'Key passes/90', 'Passes final 1/3 %']

Winger = ['Successful dribbles %', 'Goals', 'xG/90',
          'xA/90', 'Touches in box/90', 'Dribbles/90', 'Passes to penalty area/90', 'Key passes/90',
          'Progressive runs/90', 'Crosses/90', 'Deep completed crosses/90',
          'Offensive duels/90', 'PAdj Interceptions']

Forward = ['Non-penalty goals', 'xG', 'Shots on target, %', 'Goal conversion, %',
           'Dribbles/90', 'Deep completions/90', 'Passes penalty area %', 
           'Aerial duels %', 'Aerial duels/90', 'Touches in box/90']

################################################################################################################################################
#--------------------------------------------------- PERFORMANCE --------------------------------------------------------------------------------
################################################################################################################################################

def buildUpPasses(club, data):
    from datetime import timedelta

    cols = df.columns

    if data == 'WyScout':
        teamDF = df.loc[df['team.name'] == club].reset_index(drop=True)

        passesBuildUp = pd.DataFrame(columns=cols)

        contador = 0

        for idx, row in teamDF.iterrows():
            if (row['type.primary'] == 'goal_kick') & (row['pass.accurate'] == True):
                tempo = row['matchTimestamp']
                jogadas = teamDF.loc[(teamDF['matchTimestamp'] > tempo) & (teamDF['matchTimestamp'] <= timedelta(seconds=15) + tempo)]
                for i in jogadas.index.unique():
                    if (df.iloc[i]['pass.accurate'] != 'NaN'):
                        if contador == 0:
                            contador = 1
                            eventsGK = pd.DataFrame([row[cols].values], columns=cols)
                            passesBuildUp = pd.concat([passesBuildUp, eventsGK], ignore_index=True)

                        eventsGK = pd.DataFrame([jogadas.loc[i][cols].values], columns=cols)
                        passesBuildUp = pd.concat([passesBuildUp, eventsGK], ignore_index=True)
                        
                contador = 0        

        return passesBuildUp

    elif data == 'WhoScored':
        teamDF = df.loc[(df['team'] == club)].reset_index(drop=True)

        passesBuildUp = pd.DataFrame(columns=cols)

        contador = 0
        
        for idx, row in teamDF.iterrows():
            if (row['qualifiers'].__contains__('GoalKick') == True):
                tempo = row['matchTimestamp']
                jogadas = teamDF.loc[(teamDF['matchTimestamp'] > tempo) & (teamDF['matchTimestamp'] <= timedelta(seconds=15) + tempo)]
                for i in jogadas.index.unique():
                    if (df.iloc[i]['typedisplayName'] == 'Pass'):
                        if contador == 0:
                            contador = 1
                            eventsGK = pd.DataFrame([row[cols].values], columns=cols)
                            passesBuildUp = pd.concat([passesBuildUp, eventsGK], ignore_index=True)

                        eventsGK = pd.DataFrame([jogadas.loc[i][cols].values], columns=cols)
                        passesBuildUp = pd.concat([passesBuildUp, eventsGK], ignore_index=True)
                        
                contador = 0        

        return passesBuildUp

################################################################################################################################################

def carry(team, gameDay, carrydf=None, progressive=None):
    def checkCarryPositions(endX, endY, nextX, nextY):
        distance = np.sqrt(np.square(nextX - endX) + np.square(nextY - endY))
        if distance < 3:
            return True
        else:
            return False

    def isProgressiveCarry(x, y, endX, endY):
        distanceInitial = np.sqrt(np.square(105 - x) + np.square(34 - y))
        distanceFinal = np.sqrt(np.square(105 - endX) + np.square(34 - endY))
        if x < 52.5 and endX < 52.5 and distanceInitial - distanceFinal > 12.5:
            return True
        elif x < 52.5 and endX > 52.5 and distanceInitial - distanceFinal > 7.5:
            return True
        elif x > 52.5 and endX > 52.5 and distanceInitial - distanceFinal > 5:
            return True

        return False

    def get_carries(new_df, teamId):
        df = new_df.copy()
        df["recipient"] = df["playerId"].shift(-1)
        df["nextTeamId"] = df["teamId"].shift(-1)

        a = np.array(
            df[(df["typedisplayName"] == "Pass") & (df["outcomeTypedisplayName"] == "Successful") & (df["teamId"] == int(teamId))].index.tolist()
        )
        b = np.array(
            df[
                (
                    (df["typedisplayName"] == "BallRecovery")
                    | (df["typedisplayName"] == "Interception")
                    | (df["typedisplayName"] == "Tackle")
                    | (df["typedisplayName"] == "BlockedPass")
                )
                & (df["outcomeTypedisplayName"] == "Successful")
                & (df["teamId"] == int(teamId))
            ].index.tolist()
        )

        carries_df = pd.DataFrame()

        for value in a:
            carry = pd.Series()
            carry["minute"] = df.iloc[value].minute
            carry["second"] = df.iloc[value].second
            carry["playerId"] = df.iloc[value].recipient
            carry["x"] = df.iloc[value].endX
            carry["y"] = df.iloc[value].endY
            if (
                df.iloc[value + 1].typedisplayName == "OffsideGiven"
                or df.iloc[value + 1].typedisplayName == "End"
                or df.iloc[value + 1].typedisplayName == "SubstitutionOff"
                or df.iloc[value + 1].typedisplayName == "SubstitutionOn"
            ):
                continue
            elif (
                df.iloc[value + 1].typedisplayName == "Challenge"
                and df.iloc[value + 1].outcomeTypedisplayName == "Unsuccessful"
                and df.iloc[value + 1].teamId != teamId
            ):
                carry["playerId"] = df.iloc[value + 2].playerId
                value += 1
                while (df.iloc[value + 1].typedisplayName == "TakeOn" and df.iloc[value + 1].outcomeTypedisplayName == "Successful") or (
                    df.iloc[value + 1].typedisplayName == "Challenge" and df.iloc[value + 1].outcomeTypedisplayName == "Unsuccessful"
                ):
                    value += 1
                if (
                    df.iloc[value + 1].typedisplayName == "OffsideGiven"
                    or df.iloc[value + 1].typedisplayName == "End"
                    or df.iloc[value + 1].typedisplayName == "SubstitutionOff"
                    or df.iloc[value + 1].typedisplayName == "SubstitutionOn"
                ):
                    continue
            if df.iloc[value + 1].teamId != int(teamId):
                continue
            else:
                carry["endX"] = df.iloc[value + 1].x
                carry["endY"] = df.iloc[value + 1].y
            carries_df = carries_df.append(carry, ignore_index=True)

        for value in b:
            carry = pd.Series()
            carry["playerId"] = df.iloc[value].playerId
            carry["minute"] = df.iloc[value].minute
            carry["second"] = df.iloc[value].second
            carry["x"] = df.iloc[value].x
            carry["y"] = df.iloc[value].y
            if (
                df.iloc[value + 1].typedisplayName == "OffsideGiven"
                or df.iloc[value + 1].typedisplayName == "End"
                or df.iloc[value + 1].typedisplayName == "SubstitutionOff"
                or df.iloc[value + 1].typedisplayName == "SubstitutionOn"
            ):
                continue
            elif (
                df.iloc[value + 1].typedisplayName == "Challenge"
                and df.iloc[value + 1].outcomeTypedisplayName == "Unsuccessful"
                and df.iloc[value + 1].teamId != teamId
            ):
                carry["playerId"] = df.iloc[value + 2].playerId
                value += 1
                while (df.iloc[value + 1].typedisplayName == "TakeOn" and df.iloc[value + 1].outcomeTypedisplayName == "Successful") or (
                    df.iloc[value + 1].typedisplayName == "Challenge" and df.iloc[value + 1].outcomeTypedisplayName == "Unsuccessful"
                ):
                    value += 1
                if (
                    df.iloc[value + 1].typedisplayName == "OffsideGiven"
                    or df.iloc[value + 1].typedisplayName == "End"
                    or df.iloc[value + 1].typedisplayName == "SubstitutionOff"
                    or df.iloc[value + 1].typedisplayName == "SubstitutionOn"
                ):
                    continue
            if df.iloc[value + 1].playerId != df.iloc[value].playerId or df.iloc[value + 1].teamId != int(teamId):
                continue
            carry["endX"] = df.iloc[value + 1].x
            carry["endY"] = df.iloc[value + 1].y
            carries_df = carries_df.append(carry, ignore_index=True)

        carries_df["Removable"] = carries_df.apply(
            lambda row: checkCarryPositions(row["x"], row["y"], row["endX"], row["endY"]), axis=1
        )
        carries_df = carries_df[carries_df["Removable"] == False]
        return carries_df

    def isProgressivePass(x, y, endX, endY):
        distanceInitial = np.sqrt(np.square(105 - x) + np.square(34 - y))
        distanceFinal = np.sqrt(np.square(105 - endX) + np.square(34 - endY))
        if x <= 52.5 and endX <= 52.5:
            if distanceInitial - distanceFinal > 30:
                return True
        elif x <= 52.5 and endX > 52.5:
            if distanceInitial - distanceFinal > 15:
                return True
        elif x > 52.5 and endX > 52.5:
            if distanceInitial - distanceFinal > 10:
                return True
        return False

    def clean_df(df, homeTeam, awayTeam, teamId):
        names = df[["name", "playerId"]].dropna().drop_duplicates()
        df["x"] = df["x"] * 1.05
        df["y"] = df["y"] * 0.68
        df["endX"] = df["endX"] * 1.05
        df["endY"] = df["endY"] * 0.68
        df["progressive"] = False
        df["progressive"] = df[df["typedisplayName"] == "Pass"].apply(
            lambda row: isProgressivePass(row.x, row.y, row.endX, row.endY), axis=1
        )
        carries_df = get_carries(df, teamId)
        carries_df["progressiveCarry"] = carries_df.apply(
            lambda row: isProgressiveCarry(row.x, row.y, row.endX, row.endY), axis=1
        )
        carries_df["typedisplayName"] = "Carry"
        carries_df["teamId"] = teamId
        carries_df = carries_df.join(names.set_index("playerId"), on="playerId")
        df = pd.concat(
            [
                df,
                carries_df[
                    [
                        "playerId",
                        "minute",
                        "second",
                        "teamId",
                        "x",
                        "y",
                        "endX",
                        "endY",
                        "progressiveCarry",
                        "typedisplayName",
                        "name",
                    ]
                ],
            ]
        )
        df["homeTeam"] = homeTeam
        df["awayTeam"] = awayTeam
        df = df.sort_values(["minute", "second"], ascending=[True, True])
        return df

    df = df.loc[df.Match_ID == gameDay].reset_index(drop=True)
    homeTeam = df.home_Team.unique()
    homeTeam = homeTeam[0]
    awayTeam = df.away_Team.unique()
    awayTeam = awayTeam[0]

    teamID = df.loc[df.team == team].reset_index(drop=True)
    teamID = teamID.teamId.unique()
    teamID = teamID[0]

    data = clean_df(df, homeTeam, awayTeam, teamID)

    def get_progressive_carries(df, team_id):
        df_copy = df[df["teamId"] == team_id].copy()

        df_copy = df_copy[(df_copy["typedisplayName"] == "Carry") & (df_copy["progressiveCarry"] == True)]

        ret_df = df_copy.groupby(["name", "playerId"]).agg(prog_carries=("progressiveCarry", "count")).reset_index()

        return ret_df
    
    if progressive != None:
        return get_progressive_carries(data, teamID)
    elif carrydf !=None:
        return get_carries(df, teamID)
    else:
        return clean_df(df, homeTeam, awayTeam, teamID)

################################################################################################################################################

def shotAfterRecover(team):
    def recoverShot(df, team, gameDay):
        from datetime import timedelta

        cols = ['name', 'matchTimestamp', 'team', 'typedisplayName', 'x', 'y', 'away_Team', 'home_Team', 'Match_ID']

        teamDF = df.loc[(df['team'] == team) & (df['Match_ID'] ==  gameDay)].reset_index(drop=True)

        recovery_list = pd.DataFrame(columns=cols)

        contador = 0

        for idx, row in teamDF.iterrows():
            if ('BallRecovery' in row['typedisplayName']):
                tempo = row['matchTimestamp']
                jogadas = teamDF.loc[(teamDF['matchTimestamp'] > tempo) & (teamDF['matchTimestamp'] <= timedelta(seconds=10) + tempo)]
                for i in jogadas.index.unique():
                    if ('Goal' in jogadas.loc[i]['typedisplayName']):
                        if contador == 0:
                            contador = 1
                            eventsGK = pd.DataFrame([row[cols].values], columns=cols)
                            recovery_list = pd.concat([recovery_list, eventsGK], ignore_index=True)

                        eventsGK = pd.DataFrame([jogadas.loc[i][cols].values], columns=cols)
                        recovery_list = pd.concat([recovery_list, eventsGK], ignore_index=True)
                    else:
                        pass
                        
                contador = 0
                
        recovery_list = recovery_list.loc[~recovery_list.index.duplicated(), :]

        #recovery_list.drop_duplicates(inplace=True)

        return recovery_list

    def shotRecover(df, team):
        matchId = df.Match_ID.unique()
        dataAppend = []
        for id in matchId:
            data = recoverShot(df, team, id)
            dataAppend.append(data)
            
        dataAppend = pd.concat(dataAppend)
        dataAppend.reset_index(drop=True, inplace=True)
        return dataAppend
    
    return shotRecover(df, team)

################################################################################################################################################

def counterPress(team, source):
    def lost_RecoverWhoScored(df, team, gameDay):
        from datetime import timedelta

        cols = ['name', 'matchTimestamp', 'team', 'typedisplayName', 'x', 'y']

        teamDF = df.loc[(df['team'] == team) & (df['Match_ID'] == gameDay)].reset_index(drop=True)

        recovery_list = pd.DataFrame(columns=cols)

        contador = 0

        for idx, row in teamDF.iterrows():
            if ('Dispossessed' in row['typedisplayName']):
                tempo = row['matchTimestamp']
                jogadas = teamDF.loc[(teamDF['matchTimestamp'] > tempo) & (teamDF['matchTimestamp'] <= timedelta(seconds=5) + tempo)]
                for i in jogadas.index.unique():
                    if ('BallRecovery' in jogadas.loc[i]['typedisplayName']):
                        if contador == 0:
                            contador = 1
                            eventsGK = pd.DataFrame([row[cols].values], columns=cols)
                            recovery_list = pd.concat([recovery_list, eventsGK], ignore_index=True)

                        eventsGK = pd.DataFrame([jogadas.loc[i][cols].values], columns=cols)
                        recovery_list = pd.concat([recovery_list, eventsGK], ignore_index=True)
                        
                contador = 0
                
        recovery_list = recovery_list.loc[~recovery_list.index.duplicated(), :]
        #recovery_list.drop_duplicates(inplace=True)

        return recovery_list

    def lost_Recover(df, team, gameDay):
        from datetime import timedelta
        cols = ['player.name', 'matchTimestamp', 'team.name', 'type.secondary', 'location.x', 'location.y']

        teamDF = df.loc[(df['team.name'] == team) & (df['Match_ID'] == gameDay)].reset_index(drop=True)

        recovery_list = pd.DataFrame(columns=cols)

        contador = 0

        for idx, row in teamDF.iterrows():
            if ('loose_ball_duel' in row['type.secondary']) & ('recovery' not in row['type.secondary']):
                tempo = row['matchTimestamp']
                jogadas = teamDF.loc[(teamDF['matchTimestamp'] > tempo) & (teamDF['matchTimestamp'] <= timedelta(seconds=10) + tempo)]
                for i in jogadas.index.unique():
                    if ('counterpressing_recovery' in jogadas.loc[i]['type.secondary']):
                        if contador == 0:
                            contador = 1
                            eventsGK = pd.DataFrame([row[cols].values], columns=cols)
                            recovery_list = pd.concat([recovery_list, eventsGK], ignore_index=True)

                        eventsGK = pd.DataFrame([jogadas.loc[i][cols].values], columns=cols)
                        recovery_list = pd.concat([recovery_list, eventsGK], ignore_index=True)
                        
                contador = 0
                
        recovery_list = recovery_list.loc[~recovery_list.index.duplicated(), :]

        return recovery_list

    def lostRecoverAllGames(df, team):
        matchId = df.Match_ID.unique()
        dataAppend = []
        for id in matchId:
            if source == 'WyScout':
                data = lost_Recover(df, team, id)
                dataAppend.append(data)
            elif source == 'WhoScored':
                data = lost_RecoverWhoScored(df, team, id)
                dataAppend.append(data)
            
        dataAppend = pd.concat(dataAppend)
        dataAppend.reset_index(drop=True, inplace=True)

        return dataAppend
    
    return lostRecoverAllGames(df, team)

################################################################################################################################################

def defensiveCoverList(data):
    
    contador = 0

    if data == 'WyScout':
        cols = ['player.name', 'team.name', 'matchTimestamp', 'type.secondary', 'location.x', 'location.y']

        defensiveCover_list = pd.DataFrame(columns=cols)
        for idx, row in df.iterrows():
            if (row['groundDuel.duelType'] == 'dribble'):
                if ('recovery' in df.iloc[idx+1]['type.secondary']):
                    if contador == 0:
                        contador = 1
                        eventsGK = pd.DataFrame([row[cols].values], columns=cols)
                        defensiveCover_list = pd.concat([defensiveCover_list, eventsGK], ignore_index=True)

                    eventsGK = pd.DataFrame([df.iloc[idx+1][cols].values], columns=cols)
                    defensiveCover_list = pd.concat([defensiveCover_list, eventsGK], ignore_index=True)
                    
            contador = 0
            
        defensiveCover_list = defensiveCover_list.loc[~defensiveCover_list.index.duplicated(), :]
        
    elif data == 'WhoScored':
        cols = ['name', 'team', 'expandedMinute', 'typedisplayName', 'qualifiers', 'x', 'y']

        defensiveCover_list = pd.DataFrame(columns=cols)
        for idx, row in df.iterrows():
            if (row['typedisplayName'] == 'TakeOn'):
                if ('BallRecovery' in df.iloc[idx+1]['typedisplayName']):
                    if contador == 0:
                        contador = 1
                        eventsGK = pd.DataFrame([row[cols].values], columns=cols)
                        defensiveCover_list = pd.concat([defensiveCover_list, eventsGK], ignore_index=True)

                    eventsGK = pd.DataFrame([df.iloc[idx+1][cols].values], columns=cols)
                    defensiveCover_list = pd.concat([defensiveCover_list, eventsGK], ignore_index=True)
                    
            contador = 0
            
        defensiveCover_list = defensiveCover_list.loc[~defensiveCover_list.index.duplicated(), :]
    #defensiveCover_list.drop_duplicates(inplace=True)

    return defensiveCover_list

################################################################################################################################################

def cluster_Event(data, teamName, event_name, n_clusters, dataSource):

  if dataSource == 'WhoScored':
    cols_Cluster = ['team', 'typedisplayName', 'qualifiers', 'x', 'y', 'endX', 'endY']

    cols_coords = ['x', 'y', 'endX', 'endY']

    df_cluster = data[cols_Cluster]

    df_cluster = df_cluster.loc[(df_cluster['team'] == teamName) & (df_cluster['qualifiers'].str.contains(event_name) == True)].reset_index(drop=True)

    X = np.array(df_cluster[cols_coords])
    kmeans = KMeans(n_clusters = n_clusters, random_state=100)
    kmeans.fit(X)
    df_cluster['cluster'] = kmeans.predict(X)
    
  return df_cluster

################################################################################################################################################

def cluster_Shots(data, teamName, n_clusters):

  if data == 'WyScout':
    cols_Cluster = ['team.name', 'player.name', 'type.primary', 'type.secondary', 'location.x', 'location.y', 'pass.endLocation.x', 'pass.endLocation.y', 'possession.endLocation.x', 'possession.endLocation.y', 'shot.xg', 'shot.postShotXg', 'shot.isGoal']

    cols_coords = ['location.x', 'location.y', 'pass.endLocation.x', 'pass.endLocation.y', 'possession.endLocation.x', 'possession.endLocation.y']

    df_cluster = df[cols_Cluster]

    df_cluster = df_cluster.loc[(df_cluster['team.name'] == teamName) & (df['possession.attack.xg'] >= 0.12) &
                                (df['possession.attack.withShot'] == True)].reset_index(drop=True)

    X = np.array(df_cluster[cols_coords])
    kmeans = KMeans(n_clusters = n_clusters, random_state=100)
    kmeans.fit(X)
    df_cluster['cluster'] = kmeans.predict(X)
  
  elif data == 'WhoScored':
    cols_Cluster = ['team', 'player', 'typedisplayName', 'x', 'y', 'endX', 'endY', 'isShot', 'isGoal']

    cols_coords = ['x', 'y', 'endX', 'endY']

    df_cluster = df[cols_Cluster]

    df_cluster = df_cluster.loc[(df_cluster['team'] == teamName) & (df['isShot'] == True)].reset_index(drop=True)

    X = np.array(df_cluster[cols_coords])
    kmeans = KMeans(n_clusters = n_clusters, random_state=100)
    kmeans.fit(X)
    df_cluster['cluster'] = kmeans.predict(X)
    
  return df_cluster

################################################################################################################################################

def sides(xTDF, data, club):

    if data == 'WyScout':
        xTDF = xTDF.loc[(xTDF['team.name'] == club)].reset_index(drop=True)

        left_xT = xTDF[(xTDF['location.y'] >= 67) & (xTDF['location.x'] >= 55)]
        left_xT['side'] = 'Left'

        center_xT = xTDF[(xTDF['location.y'] < 67) & (xTDF['location.y'] > 33) & (xTDF['location.x'] >= 55)]
        center_xT['side'] = 'Center'

        right_xT = xTDF[(xTDF['location.y'] <= 33) & (xTDF['location.x'] >= 55)]
        right_xT['side'] = 'Right'

        sides = pd.concat([left_xT, center_xT, right_xT], axis=0)
        
        return sides
    
    elif data == 'WhoScored':
        xTDF = xTDF.loc[(xTDF['team'] == club)].reset_index(drop=True)

        left_xT = xTDF[(xTDF['y'] >= 67) & (xTDF['x'] >= 55)]
        left_xT['side'] = 'Left'

        center_xT = xTDF[(xTDF['y'] < 67) & (xTDF['y'] > 33) & (xTDF['x'] >= 55)]
        center_xT['side'] = 'Center'

        right_xT = xTDF[(xTDF['y'] <= 33) & (xTDF['x'] >= 55)]
        right_xT['side'] = 'Right'

        sides = pd.concat([left_xT, center_xT, right_xT], axis=0)

        return sides
    
################################################################################################################################################

def dataFrame_xTFlow():

    leftfinal3rd = []
    centerfinal3rd = []
    rightfinal3rd = []

    left = df.loc[(df['side'] == 'Left'), 'xT'].sum()
    center = df.loc[(df['side'] == 'Center'), 'xT'].sum()
    right = df.loc[(df['side'] == 'Right'), 'xT'].sum()
    
    leftfinal3rd.append(left)
    centerfinal3rd.append(center)
    rightfinal3rd.append(right)

    data = {
        'left_xT' : leftfinal3rd,
        'center_xT' : centerfinal3rd,
        'right_xT' : rightfinal3rd
    }
    
    df = pd.DataFrame(data)
    
    return df

################################################################################################################################################

def dataFramexTFlow(dataDF, club, dataSource):

    if dataSource == 'WyScout':
        df_Home = dataDF.loc[(dataDF['team.name'] == club)].reset_index(drop=True)

        df_Away = dataDF.loc[(dataDF['team.name'] != club)].reset_index(drop=True)
        
    elif dataSource == 'WhoScored':
        df_Home = dataDF.loc[(dataDF['team'] == club)].reset_index(drop=True)

        df_Away = dataDF.loc[(dataDF['team'] != club)].reset_index(drop=True)
        
    home_xT = []
    away_xT = []

    #Criação da lista de jogadores
    Minutes = range(dataDF['minute'].min(), dataDF['minute'].max())

    #Ciclo For de atribuição dos valores a cada jogador
    for minute in Minutes:
        home_xT.append(df_Home.loc[df_Home['minute'] == minute, 'xT'].sum())
        away_xT.append(df_Away.loc[df_Away['minute'] == minute, 'xT'].sum())
        
    data = {
        'Minutes' : Minutes,
        'home_xT' : home_xT,
        'away_xT' : away_xT
        }

    dataDF = pd.DataFrame(data)
    return dataDF

################################################################################################################################################

def dataFrame_touchFlow(team):

    df_Home = df.loc[(df['team.name'] == team) & (df['location.x'] >= 78) & (df['pass.accurate'] == True)]

    df_Away = df.loc[(df['team.name'] != team) & (df['location.x'] >= 78) & (df['pass.accurate'] == True)]

    home_Team = df_Home['team.name'].unique()

    home_Team[0]

    away_Team = df_Away['team.name'].unique()

    away_Team[0]

    goal_Home = df.loc[(df['team.name'] == team) & (df['shot.isGoal'] == True)]

    goal_Away = df.loc[(df['team.name'] != team) & (df['shot.isGoal'] == True)]

    home_Touches = []
    away_Touches = []

    mini = df['minute'].min()
    maxi = df['minute'].max()

    #Criação da lista de jogadores
    Minutes = range(mini, maxi)

    #Ciclo For de atribuição dos valores a cada jogador
    for minute in Minutes:
        home_Touches.append(df_Home.loc[df_Home['minute'] == minute, 'pass.accurate'].count())
        away_Touches.append(df_Away.loc[df_Away['minute'] == minute, 'pass.accurate'].count())
    data = {
        'Minutes' : Minutes,
        'Home' : home_Team[0],
        'Away' : away_Team[0],
        'Goal_Home' : len(goal_Home),
        'Goal_Away' : len(goal_Away),
        'home_Touches' : home_Touches,
        'away_Touches' : away_Touches
        }

    df = pd.DataFrame(data)
    
    return df

################################################################################################################################################

def search_qualifierOPTA(list_Name, event):
  cols = df.columns

  list_Name = pd.DataFrame(columns=cols)

  for idx, row in df.iterrows():
    if event in row['qualifiers']:
        events = pd.DataFrame([df.iloc[idx][cols].values], columns=cols)
        list_Name = pd.concat([list_Name, events], ignore_index=True)
          
  list_Name = list_Name.loc[~list_Name.index.duplicated(), :]

  return list_Name

################################################################################################################################################

def xT(data, dataSource):
  eventsPlayers_xT = df.copy()

  #Import xT Grid, turn it into an array, and then get how many rows and columns it has
  xT = pd.read_csv('C:/Users/menes/Documents/Data Hub/xT/xT_Grid.csv', header=None)
  xT = np.array(xT)
  xT_rows, xT_cols = xT.shape


  if dataSource == 'WyScout':
    eventsPlayers_xT['x1_bin'] = pd.cut(eventsPlayers_xT['location.x'], bins=xT_cols, labels=False)
    eventsPlayers_xT['y1_bin'] = pd.cut(eventsPlayers_xT['location.y'], bins=xT_rows, labels=False)
    eventsPlayers_xT['x2_bin'] = pd.cut(eventsPlayers_xT['pass.endLocation.x'], bins=xT_cols, labels=False)
    eventsPlayers_xT['y2_bin'] = pd.cut(eventsPlayers_xT['pass.endLocation.y'], bins=xT_rows, labels=False)

    eventsPlayers_xT = eventsPlayers_xT[['player.name', 'team.name', 'minute', 'second', 'location.x', 'location.y', 'type.primary', 'type.secondary', 'pass.endLocation.x', 'pass.endLocation.y', 'x1_bin', 'y1_bin', 'x2_bin', 'y2_bin']]

    eventsPlayers_xT['start_zone_value'] = eventsPlayers_xT[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    eventsPlayers_xT['end_zone_value'] = eventsPlayers_xT[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)

    eventsPlayers_xT['xT'] = round(eventsPlayers_xT['end_zone_value'] - eventsPlayers_xT['start_zone_value'], 2)

    eventsPlayers_xT.drop(eventsPlayers_xT.index[0], axis=0, inplace=True)

    eventsPlayers_xT.reset_index(inplace=True)

    eventsPlayers_xT.drop(['index'], axis=1, inplace=True)
    
  elif dataSource == 'WhoScored':
    eventsPlayers_xT['x1_bin'] = pd.cut(eventsPlayers_xT['x'], bins=xT_cols, labels=False)
    eventsPlayers_xT['y1_bin'] = pd.cut(eventsPlayers_xT['y'], bins=xT_rows, labels=False)
    eventsPlayers_xT['x2_bin'] = pd.cut(eventsPlayers_xT['endX'], bins=xT_cols, labels=False)
    eventsPlayers_xT['y2_bin'] = pd.cut(eventsPlayers_xT['endY'], bins=xT_rows, labels=False)

    eventsPlayers_xT = eventsPlayers_xT[['id', 'Match_ID', 'matchTimestamp', 'outcomeTypedisplayName', 'name', 'team', 'minute', 'second', 'x', 'y', 'typedisplayName', 'isTouch', 'endX', 'endY', 'x1_bin', 'y1_bin', 'x2_bin', 'y2_bin']]

    eventsPlayers_xT['start_zone_value'] = eventsPlayers_xT[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    eventsPlayers_xT['end_zone_value'] = eventsPlayers_xT[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)

    eventsPlayers_xT['xT'] = round(eventsPlayers_xT['end_zone_value'] - eventsPlayers_xT['start_zone_value'], 2)

    eventsPlayers_xT.drop(eventsPlayers_xT.index[0], axis=0, inplace=True)

    eventsPlayers_xT.reset_index(inplace=True)

    eventsPlayers_xT.drop(['index'], axis=1, inplace=True)
  
  elif data == 'BePro':
    eventsPlayers_xT['x1_bin'] = pd.cut(eventsPlayers_xT['x_start'], bins=xT_cols, labels=False)
    eventsPlayers_xT['y1_bin'] = pd.cut(eventsPlayers_xT['y_start'], bins=xT_rows, labels=False)
    eventsPlayers_xT['x2_bin'] = pd.cut(eventsPlayers_xT['x_end'], bins=xT_cols, labels=False)
    eventsPlayers_xT['y2_bin'] = pd.cut(eventsPlayers_xT['y_end'], bins=xT_rows, labels=False)

    eventsPlayers_xT = eventsPlayers_xT[['id', 'Match_ID', 'matchTimestamp', 'team_name', 'player_name', 'event_time', 'x_start', 'y_start', 'eventType', 'x_end', 'y_end', 'x1_bin', 'y1_bin', 'x2_bin', 'y2_bin']]

    eventsPlayers_xT['start_zone_value'] = eventsPlayers_xT[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    eventsPlayers_xT['end_zone_value'] = eventsPlayers_xT[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)

    eventsPlayers_xT['xT'] = round(eventsPlayers_xT['end_zone_value'] - eventsPlayers_xT['start_zone_value'], 2)

    eventsPlayers_xT.drop(eventsPlayers_xT.index[0], axis=0, inplace=True)

    eventsPlayers_xT.reset_index(inplace=True)

    eventsPlayers_xT.drop(['index'], axis=1, inplace=True)

  return eventsPlayers_xT

################################################################################################################################################

def touch_Map(club, Player, gameID):

        color = clubColors.get(club)

        player_df = df.loc[(df['name'] == Player)]

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(6,4))

        pitch = Pitch(pitch_type='opta',
                      pitch_color='#E8E8E8', line_color='#181818',
                      line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        #############################################################################################################################################

        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
        [{"color": color[0], "fontweight": 'bold'}]

        if (Player == None) & (gameID != 'All Season'):
                fig_text(s =f'<{club}>' + ' ' + 'Touch Map',
                        x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center', va='center', fontsize=48);
                
                fig_text(s ='MatchDay:' + str(gameID) + ' ' +  '| Season 21-22 | @menesesp20',
                        x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=16, alpha=0.7);

        elif (Player == None) & (gameID == 'All Season'):
                fig_text(s =f'<{club}>' + ' ' + 'Touch Map',
                        x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center', va='center', fontsize=48);
                
                fig_text(s ='All Season' + ' ' +  '| Season 21-22 | @menesesp20',
                        x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=16, alpha=0.7);

        if (Player != None) & (gameID != 'All Season'):
                fig_text(s =f'<{Player}>' + ' ' + 'Touch Map',
                        x = 0.5, y = 0.93, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center', va='center', fontsize=16);
                
                fig_text(s ='MatchDay:' + str(gameID) + ' ' +  '| World Cup Catar 2022 | @menesesp20',
                        x = 0.5, y = 0.88, color='#181818', fontweight='bold', ha='center', va='center', fontsize=5, alpha=0.7);

        elif (Player != None) & (gameID == 'All Season'):
                fig_text(s =f'<{Player}>' + ' ' + 'Touch Map',
                        x = 0.5, y = 0.93, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center', va='center', fontsize=16);
                
                fig_text(s ='All Season ' +  '| World Cup Catar 2022 | @menesesp20',
                        x = 0.5, y = 0.88, color='#181818', fontweight='bold', ha='center', va='center', fontsize=5, alpha=0.7);

        #############################################################################################################################################

        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#E8E8E8', color[0]], N=10)
        bs = pitch.bin_statistic(player_df['x'], player_df['y'], bins=(8, 8))
        pitch.heatmap(bs, edgecolors='#1b1b1b', lw=0.3, ax=ax, cmap=pearl_earring_cmap, zorder=2)

        #filter that dataframe to exclude outliers. Anything over a z score of 1 will be excluded for the data points
        convex = player_df[(np.abs(stats.zscore(player_df[['x','y']])) < 1).all(axis=1)]

        hull = pitch.convexhull(convex['x'], convex['y'])

        pitch.polygon(hull, ax=ax, edgecolor='#181818', facecolor='#181818', alpha=0.5, linestyle='--', linewidth=2.5, zorder=2)

        pitch.scatter(player_df['x'], player_df['y'], ax=ax, edgecolor='#181818', facecolor='black', alpha=0.5, zorder=2)

        pitch.scatter(x=convex['x'].mean(), y=convex['y'].mean(), ax=ax, c='#E8E8E8', edgecolor=color[0], s=200, zorder=5)

        #############################################################################################################################################

        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.08, bottom=0.89, width=0.2, height=0.08)

        fig_text(s = 'Attacking Direction',
                        x = 0.5, y = 0.08,
                        color='#181818', fontweight='bold',
                        ha='center', va='center',
                        fontsize=8)

        # ARROW DIRECTION OF PLAY
        ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))
        
        plt.savefig('assets/touchMap' + Player + '.png', dpi=300)
        
        return app.get_asset_url('touchMap' + Player + '.png')

################################################################################################################################################

def heatMap_xT(club, player=None):

        color = clubColors.get(club)
        
        dfXT = df.loc[(df['typedisplayName'] == 'Pass') & (df['outcomeTypedisplayName'] == 'Successful')].reset_index(drop=True)

        xTDF = xT(dfXT, 'WhoScored')

        if (player == None):
                xTheatMap = xTDF.loc[(xTDF['team'] == club)]
        else:
                xTheatMap = xTDF.loc[(xTDF['name'] == player)]

        fig, ax = plt.subplots(figsize=(6,4))

        pitch = Pitch(pitch_type='opta',
                      pitch_color='#E8E8E8', line_color='#181818',
                      line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#E8E8E8', color[0]], N=10)

        xTheatHeat = xTheatMap.loc[xTheatMap.xT > 0]
        bs = pitch.bin_statistic(xTheatHeat['x'], xTheatHeat['y'], bins=(10, 8))
        pitch.heatmap(bs, edgecolors='#E8E8E8', ax=ax, cmap=pearl_earring_cmap)

        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.05, bottom=0.89, width=0.2, height=0.08)

        highlight_textprops =\
        [{"color": color[0], "fontweight": 'bold'}]

        # TITLE
        if player == None:
                fig_text(s = 'Where' + ' ' + f'<{club}>' + ' ' + 'generate the most xT',
                        x = 0.5, y = 0.95,  highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center' ,fontsize=14);
                
                fig_text(s ='All Season ' +  '| World Cup Catar 2022 | @menesesp20',
                        x = 0.5, y = 0.903, color='#181818', fontweight='bold', ha='center', va='center', fontsize=5, alpha=0.7);
        else:

                fig_text(s = 'Where' + ' ' + f'<{player}>' + ' ' + 'generate the xT',
                        x = 0.5, y = 0.93, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center', va='center', fontsize=14);
                
                fig_text(s ='All Season ' +  '| World Cup Catar 2022 | @menesesp20',
                        x = 0.5, y = 0.88, color='#181818', fontweight='bold', ha='center', va='center', fontsize=5, alpha=0.7);
        #fig_text(s = 'Coach: Jorge Jesus',
        #         x = 0.123, y = 0.97, color='#181818', fontweight='bold', ha='center', alpha=0.8, fontsize=12);

        # TOTAL xT
        fig_text(s = str(round(sum(xTheatMap.xT), 2)) + ' ' + 'xT Generated', 
                x = 0.51, y = 1.02, color='#181818', fontweight='bold', ha='center' ,fontsize=5);

        fig_text(s = 'Attacking Direction',
                        x = 0.5, y = 0.08,
                        color='#181818', fontweight='bold',
                        ha='center', va='center',
                        fontsize=8)

        # ARROW DIRECTION OF PLAY
        ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))
        
        if player != None:
            plt.savefig('assets/xTMap' + player + '.png', dpi=300)
            
            return app.get_asset_url('xTMap' + player + '.png')
        
        else:
            plt.savefig('assets/xTMap' + club + '.png', dpi=300)
            
            return app.get_asset_url('xTMap' + club + '.png')

################################################################################################################################################

def heatMapChances(team, data, player=None):
    
    color = clubColors.get(team)

    if data == 'WyScout':
        # Plotting the pitch
        fig, ax = plt.subplots(figsize=(6,4))

        pitch = Pitch(pitch_type='opta',
                      pitch_color='#E8E8E8', line_color='#181818',
                      line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        fig_text(s = 'Where has ' + team + ' created from',
                    x = 0.53, y = 0.94, fontweight='bold',
                    ha='center',fontsize=14, color='#181818');

        fig_text(s = 'All open-play chances created in the ' + 'World Cup Catar 2022',
                    x = 0.53, y = 0.9, fontweight='bold',
                    ha='center',fontsize=5, color='#181818', alpha=0.4);

        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + team + '.png', fig=fig, left=0.25, bottom=0.885, width=0.08, height=0.07)

        # Opportunity
        opportunity = df.loc[(df['location.x'] >= 50) & (df['team.name'] == team) & (df['type.secondary'].apply(lambda x: 'opportunity' in x))].reset_index(drop=True)

        #bin_statistic = pitch.bin_statistic_positional(opportunity['location.x'], opportunity['location.y'], statistic='count',
        #                                               positional='full', normalize=True)

        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                            ['#e8e8e8', '#3d0000', '#ff0000'], N=10)

        path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                    path_effects.Normal()]

        bin_x = np.linspace(pitch.dim.left, pitch.dim.right, num=7)
        bin_y = np.sort(np.array([pitch.dim.bottom, pitch.dim.six_yard_bottom,
                                pitch.dim.six_yard_top, pitch.dim.top]))

        bs = pitch.bin_statistic(opportunity['location.x'], opportunity['location.y'],  statistic='count', normalize=True, bins=(bin_x, 5))

        pitch.heatmap(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.5)

        pitch.label_heatmap(bs, color='#E8E8E8', fontsize=18,
                                    ax=ax, ha='center', va='center',
                                    str_format='{:.0%}', path_effects=path_eff)
        
    elif data == 'WhoScored':
        # Plotting the pitch
        fig, ax = plt.subplots(figsize=(6,4))

        pitch = Pitch(pitch_type='opta',
                      pitch_color='#E8E8E8', line_color='#181818',
                      line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        highlight_textprops =\
        [{"color": color[0], "fontweight": 'bold'}]

        if player == None:
            
            fig_text(s = 'Where has ' + f'<{team}>' + ' created from',
                     highlight_textprops=highlight_textprops,
                     x = 0.53, y = 0.95, fontweight='bold',
                     ha='center',fontsize=14, color='#181818');
            player_df = df.loc[df.team == team].reset_index(drop=True)
        
        else:
            player_df = df.loc[df.name == player].reset_index(drop=True)

            fig_text(s = 'Where has ' + player + ' created from',
                        x = 0.53, y = 0.95, fontweight='bold',
                        ha='center',fontsize=14, color='#181818');

        fig_text(s = 'All open-play chances created in the ' + 'World Cup Catar 2022',
                    x = 0.53, y = 0.9, fontweight='bold',
                    ha='center',fontsize=5, color='#181818', alpha=0.4);

        #fig_text(s = 'Coach: Jorge Jesus',
        #         x = 0.29, y = 0.862, color='#181818', fontweight='bold', ha='center', alpha=0.8, fontsize=6);

        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + team + '.png', fig=fig, left=0.15, bottom=0.90, width=0.08, height=0.07)

        # Opportunity
        opportunity = player_df.loc[(player_df['x'] >= 50) & (player_df['team'] == team) & (player_df['qualifiers'].str.contains('KeyPass') == True)].reset_index(drop=True)

        #bin_statistic = pitch.bin_statistic_positional(opportunity['x'], opportunity['y'], statistic='count',
        #                                               positional='full', normalize=True)

        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                            ['#e8e8e8', color[0]], N=10)

        path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                    path_effects.Normal()]

        bs = pitch.bin_statistic(opportunity['x'], opportunity['y'],  statistic='count', normalize=True, bins=(7, 5))

        pitch.heatmap(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.5)

        pitch.label_heatmap(bs, color='#E8E8E8', fontsize=11,
                                    ax=ax, ha='center', va='center',
                                    str_format='{:.0%}', path_effects=path_eff)
        
        plt.savefig('assets/chancesCreated' + team + '.png', dpi=300)
        
        return app.get_asset_url('chancesCreated' + team + '.png')

################################################################################################################################################

def passing_networkWhoScored(team, gameDay, afterSub=None):

        if gameDay != 'All Season':
            dataDF = df.loc[df.Match_ID == gameDay].reset_index(drop=True)
            
        elif gameDay == 'All Season':
            dataDF = df.copy()

        color = clubColors.get(team)

        data = xT(dataDF, 'WhoScored')

        ###########################################################################################################################
        if gameDay != 'All Season':
            network = data.loc[(data['team'] == team) & (data['Match_ID'] == gameDay)].reset_index(drop=True)
            
        elif gameDay == 'All Season':
            network = data.loc[(data['team'] == team)].reset_index(drop=True)
            
        network = network.sort_values(['matchTimestamp'], ascending=True)

        network["newsecond"] = 60 * network["minute"] + network["second"]

        #find time of the team's first substitution and filter the df to only passes before that
        Subs = network.loc[(network['typedisplayName'] == "SubstitutionOff")]
        SubTimes = Subs["newsecond"]
        SubOne = SubTimes.min()

        passes = network.loc[(network['typedisplayName'] == "Pass") &
                             (network['outcomeTypedisplayName'] == 'Successful')].reset_index(drop=True)


        ###########################################################################################################################
        if afterSub == None:
          passes = passes.loc[passes['newsecond'] < SubOne].reset_index(drop=True)

        elif afterSub != None:
          passes = passes.loc[passes['newsecond'] > SubOne].reset_index(drop=True)

        ###########################################################################################################################

        passes['passer'] = passes['name']
        passes['recipient'] = passes['passer'].shift(-1)
        passes['passer'] = passes['passer'].astype(str)
        passes['recipient'] = passes['recipient'].astype(str)

        passes = passes.loc[passes['recipient'] != 'nan']

        ###########################################################################################################################

        avg = passes.groupby('passer').agg({'x':['mean'], 'y':['mean', 'count']})
        avg.columns = ['x_avg', 'y_avg', 'count']

        player_pass_count = passes.groupby("passer").size().to_frame("num_passes")
        player_pass_value = passes.groupby("passer")['xT'].sum().to_frame("pass_value")

        passes["pair_key"] = passes.apply(lambda x: "_".join(sorted([x["passer"], x["recipient"]])), axis=1)
        pair_pass_count = passes.groupby("pair_key").size().to_frame("num_passes")
        pair_pass_value = passes.groupby("pair_key")['xT'].sum().to_frame("pass_value")

        ###########################################################################################################################

        btw = passes.groupby(['passer', 'recipient']).id.count().reset_index()
        btw.rename({'id':'pass_count'}, axis='columns', inplace=True)

        merg1 = btw.merge(avg, left_on='passer', right_index=True)
        pass_btw = merg1.merge(avg, left_on='recipient', right_index=True, suffixes=['', '_end'])

        pass_btw = pass_btw.loc[pass_btw['pass_count'] > 5]

        ##################################################################################################################################################################

        fig, ax = plt.subplots(figsize=(6,4))

        pitch = VerticalPitch(pitch_type='opta',
                              pitch_color='#E8E8E8', line_color='#181818',
                              line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#e8e8e8', color[0]], N=10)

        bs = pitch.bin_statistic(passes['endX'], passes['endY'], bins=(6, 3))

        pitch.heatmap(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.5)

        fig.set_facecolor('#E8E8E8')

        max_player_count = None
        max_player_value = None
        max_pair_count = None
        max_pair_value = None
        
        max_player_count = player_pass_count.num_passes.max() if max_player_count is None else max_player_count
        max_player_value = player_pass_value.pass_value.max() if max_player_value is None else max_player_value
        max_pair_count = pair_pass_count.num_passes.max() if max_pair_count is None else max_pair_count
        max_pair_value = pair_pass_value.pass_value.max() if max_pair_value is None else max_pair_value

        avg['x_avg'] = round(avg['x_avg'], 2)
        avg['y_avg'] = round(avg['y_avg'], 2)
        pair_stats = pd.merge(pair_pass_count, pair_pass_value, left_index=True, right_index=True)

        #std = mundial.loc[(mundial.isTouch ==True) & (mundial.team == 'Portugal')].reset_index(drop=True)
        #std = std.loc[std['newsecond'] < SubOne]
        #std = std.groupby('name').agg({'x':['std'], 'y':['mean']})
        #std.columns = ['x_std', 'y_mean']

        for pair_key, row in pair_stats.iterrows():
            player1, player2 = pair_key.split("_")
            
            player1_x = avg.loc[player1]["x_avg"]
            player1_y = avg.loc[player1]["y_avg"]

            player2_x = avg.loc[player2]["x_avg"]
            player2_y = avg.loc[player2]["y_avg"]

            num_passes = row["num_passes"]
            if num_passes > 3:
                    num_passes = 3
                    
            pass_value = row["pass_value"]

            norm = Normalize(vmin=0, vmax=max_pair_value)
            edge_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#181818', color[0]], N=10)
            edge_color = edge_cmap(norm(pass_value))

            ax.plot([player1_y, player2_y], [player1_x, player2_x],
                    'w-', linestyle='-', alpha=1, lw=num_passes, zorder=2, color=edge_color)

            #playerx_std = std.loc[player1]["x_std"]
            #playery_mean = std.loc[player1]["y_mean"]

            #ax.scatter(playerx_std, playery_mean, s=2, c=edge_color, marker = 'v')

        #plot arrows
        #def pass_line_template(ax, x, y, end_x, end_y, lw):
        #        ax.annotate('', xy=(end_y, end_x), xytext=(y, x), zorder=2,
        #        arrowprops=dict(arrowstyle='-|>', linewidth=lw, color='#181818', alpha=.85))

        # PLOT LINES        
        #def pass_line_template(ax, x, y, end_x, end_y, line_color):
        #        pitch.lines(x, y, end_x, end_y, lw=3, transparent=True, comet=True, cmap=line_color, ax=ax)

        #def pass_line_template_shrink(ax, x, y, end_x, end_y, lw, dist_delta=1):
        #        dist = math.hypot(end_x - x, end_y - y)
        #        angle = math.atan2(end_y-y, end_x-x)
        #        upd_x = x + (dist - dist_delta) * math.cos(angle)
        #        upd_y = y + (dist - dist_delta) * math.sin(angle)
        #        pass_line_template(ax, x, y, upd_x, upd_y, lw)
                
        
        #for index, row in pass_btw.iterrows():
        #        pass_line_template_shrink(ax, row['x_avg'], row['y_avg'], row['x_avg_end'], row['y_avg_end'], row['count'] * 0.03)

        #plot nodes
        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#181818', color[0]], N=10)
        cycles = 1
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        #axins = inset_axes(ax,
        #            width="100%",  
        #            height="5%",
        #            loc='lower center',
        #            borderpad=-5
        #           )                                                        
        plt.colorbar(ScalarMappable(cmap=pearl_earring_cmap), label='xT', orientation="horizontal", shrink=0.3, pad=0.)

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
        if gameDay != 'All Season':
            network = data.loc[(data['team'] == team) & (data['Match_ID'] == gameDay)].reset_index(drop=True)
            
        elif gameDay == 'All Season':
            network = data.loc[(data['team'] == team)].reset_index(drop=True)
            
        network = network.sort_values(['matchTimestamp'], ascending=True)

        network["newsecond"] = 60 * network["minute"] + network["second"]

        #find time of the team's first substitution and filter the df to only passes before that
        Subs = network.loc[(network['typedisplayName']=="SubstitutionOff")]
        SubTimes = Subs["newsecond"]
        SubOne = SubTimes.min()

        passes = network.loc[(network['typedisplayName'] == "Pass") &
                             (network['outcomeTypedisplayName'] == 'Successful')].reset_index(drop=True)

        ###########################################################################################################################
        if afterSub == None:
          passes = passes.loc[passes['newsecond'] < SubOne].reset_index(drop=True)

        elif afterSub != None:
          passes = passes.loc[passes['newsecond'] > SubOne].reset_index(drop=True)

        ###########################################################################################################################

        passes['passer'] = passes['name']
        passes['recipient'] = passes['passer'].shift(-1)
        passes['passer'] = passes['passer'].astype(str)
        passes['recipient'] = passes['recipient'].astype(str)
        
        passes = passes.loc[passes['recipient'] != 'nan']

        ###########################################################################################################################

        avg = passes.groupby('passer').agg({'x':['mean'], 'y':['mean', 'count']})
        avg.columns = ['x_avg', 'y_avg', 'count']

        ###########################################################################################################################

        btw = passes.groupby(['passer', 'recipient']).id.count().reset_index()
        btw.rename({'id':'pass_count'}, axis='columns', inplace=True)

        merg1 = btw.merge(avg, left_on='passer', right_index=True)
        pass_btw = merg1.merge(avg, left_on='recipient', right_index=True, suffixes=['', '_end'])

        pass_btw = pass_btw.loc[pass_btw['pass_count'] > 5]

        avg = pd.DataFrame(passes.groupby('passer').agg({'x':['mean'], 'y':['mean', 'count']})).reset_index()

        avg.to_excel('avgWhoScoredmundial.xlsx')

        avg = pd.read_excel('avgWhoScoredmundial.xlsx')

        avg.drop(avg.index[0:2], inplace=True)

        avg.reset_index(drop=True)

        avg.rename({'Unnamed: 4':'count'}, axis=1, inplace=True)

        avg.drop(['Unnamed: 0'], axis=1, inplace=True)

        #Criação da lista de jogadores

        test = xT(data, 'WhoScored')
        
        test = test.loc[test['team'] == team].reset_index(drop=True)
        
        players = test['name'].unique()


        players_xT = []

        #Ciclo For de atribuição dos valores a cada jogador
        for player in players:
                players_xT.append(test.loc[test['name'] == player, 'xT'].sum())
        data = {
        'passer' : players,
        'xT' : players_xT
        }

        test = pd.DataFrame(data)

        #test.drop(test.index[11], inplace=True)

        avg = pd.merge(avg, test, on='passer')

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
        pass_nodes = pitch.scatter(avg['x'], avg['y'], s=350,
                                cmap=pearl_earring_cmap, edgecolors="#010101", c=avg['xT'], linewidth=1.3, ax=ax, zorder=3)


        #Uncomment these next two lines to get each node labeled with the player id. Check to see if anything looks off, and make note of each player if you're going to add labeles later like their numbers
        for index, row in avg.iterrows():
                pitch.annotate(row.passer, xy=(row['x'], row['y']), c='#E8E8E8', va='center', ha='center', size=3, fontweight='bold', ax=ax)


        ##################################################################################################################################################################

        homeTeam = dataDF.home_Team.unique()
        homeTeam = homeTeam[0]

        awayTeam = dataDF.away_Team.unique()
        awayTeam = awayTeam[0]

        homeName = homeTeam
        color = [color[0], color[1]]

        awayName = awayTeam
        color2c = clubColors.get(awayTeam)
        color2 = [color2c[0], color2c[1]]

        ##################################################################################################################################################################

        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
                [{"color": color[0],"fontweight": 'bold'},
                {"color": color2[0],"fontweight": 'bold'}]

        fig_text(s = f'<{homeName}>' + ' ' + 'vs' + ' ' + f'<{awayName}>',
                x = 0.52, y = 0.94,
                color='#181818', fontweight='bold', ha='center',
                highlight_textprops = highlight_textprops,
                fontsize=18);

        matchID = network.Match_ID.unique()
        matchID = matchID[0]

        fig_text(s = 'Passing Network' + ' ' + '|' + ' ' + 'MatchDay' + ' ' + str(matchID) + '| World Cup Catar 2022 | @menesesp20',
                x = 0.52, y = 0.91,
                color='#181818', fontweight='bold', ha='center',
                fontsize=5);

        fig_text(s = 'The color of the nodes is based on xT value',
                 x = 0.44, y = 0.875,
                 color='#181818', fontweight='bold', ha='center',
                 fontsize=5);

        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + team + '.png', fig=fig, left=0.383, bottom=0.898, width=0.04, height=0.05)
        
        plt.savefig('assets/passingNetwork' + team + '.png', dpi=300)
        
        return app.get_asset_url('passingNetwork' + team + '.png')

################################################################################################################################################

def highTurnovers(club, gameDay, data, player=None):
    
    if data == 'WhoScored':
        
        if player == None:
            dataDF = df.loc[df.Match_ID == gameDay].reset_index(drop=True)
        else:
            dataDF = df.loc[df.name == player].reset_index(drop=True)
            
        #Plotting the pitch
        highTurnover = dataDF.loc[(dataDF['typedisplayName'] == 'BallRecovery') & (dataDF.y >= 65) & (dataDF.team == club)].reset_index(drop=True)
        highTurnover.drop_duplicates(['name', 'typedisplayName', 'x', 'y'], keep='first')

        dfShot = shotAfterRecover(club)
        dfShot = dfShot.loc[dfShot.y >= 50].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(6,4))

        pitch = VerticalPitch(pitch_type='opta',
                              pitch_color='#E8E8E8', line_color='#181818',
                              line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)
        
        fig.set_facecolor('#E8E8E8')

        #Title of our plot

        fig.suptitle(club, fontsize=14, color='black', y=0.98)

        fig_text(s = 'High Turnovers | World Cup Catar 2022 | @menesesp20',
                x = 0.5, y = 0.92, color='black', ha='center', fontsize=5);
        
        #fig_text(s = 'Coach: Jorge Jesus',
        #         x = 0.22, y = 0.86, alpha=0.8, color='black', ha='center', fontsize=12);
        
        ax.axhline(65,c='#ff0000', ls='--', lw=4)

        ax.scatter(highTurnover.x, highTurnover.y, label = 'High Turnovers' + ' ' + '(' + f'{len(highTurnover)}' + ')',
                            c='#ff0000', marker='o', edgecolor='#181818', s=25, zorder=5)
        
        ax.scatter(dfShot.x, dfShot.y, label = 'Shot after a turnover within 5 seconds' + ' ' + '(' + f'{len(dfShot)}' + ')',
                            c='#ffba08', marker='*', edgecolor='#181818', s=50, zorder=5)

        #Criação da legenda
        l = ax.legend(bbox_to_anchor=(0.04, 0.3), loc='upper left', facecolor='white', framealpha=0, labelspacing=.8, prop={'size': 4})
        
        #Ciclo FOR para atribuir a white color na legend
        for text in l.get_texts():
            text.set_color('#181818')

        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.15, bottom=0.895, width=0.2, height=0.09)

        if player != None:
            # Player Image
            fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '/' + player + '.png', fig=fig, left=0.15, bottom=0.846, width=0.08, height=0.06)
        
        add_image(image='Images/WorldCup_Qatar.png', fig=fig, left=0.75, bottom=0.895, width=0.08, height=0.1)
        
        plt.savefig('assets/turnOvers' + club + '.png', dpi=300)
        
        return app.get_asset_url('turnOvers' + club + '.png')

################################################################################################################################################

def draw_heatmap_construcao(club, data, player=None):

  color = clubColors.get(club)

  passesGk = buildUpPasses(club, data)

  fig, ax = plt.subplots(figsize=(6,4))

  pitch = Pitch(pitch_type='opta',
                        pitch_color='#E8E8E8', line_color='#181818',
                        line_zorder=3, linewidth=0.5, spot_scale=0.00)

  pitch.draw(ax=ax)
  
  fig.set_facecolor('#e8e8e8')

  pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                        ['#e8e8e8', color[0]], N=10)

  if data == 'WyScout':
    passesGk['location.x'] = passesGk['location.x'].astype(float)
    passesGk['location.y'] = passesGk['location.y'].astype(float)
  
    bs = pitch.bin_statistic(passesGk['location.x'], passesGk['location.y'], bins=(12, 8))
  
  elif data == 'WhoScored':
    passesGk['x'] = passesGk['x'].astype(float)
    passesGk['y'] = passesGk['y'].astype(float)

    bs = pitch.bin_statistic(passesGk['x'], passesGk['y'], bins=(12, 8))

  pitch.heatmap(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap)

  fig_text(s = "How do they come out playing",
          x = 0.5, y = 0.97,
          color='#181818', ha='center', fontsize=14);

  fig_text(s = "GoalKick | World Cup 2022 | @menesesp20",
          x = 0.5, y = 0.93,
          color='#181818', ha='center', fontsize=5);

  #fig_text(s = "Coach: Roger Schmidt",
  #        x = 0.21, y = 0.88,
  #        color='#181818', ha='center', alpha=0.8, fontsize=14);

  # Club Logo
  fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.05, bottom=0.89, width=0.15, height=0.1)

  fig_text(s = 'Attacking Direction',
           x = 0.5, y = 0.1,
           color='#181818',
           ha='center', va='center',
           fontsize=8)

  # ARROW DIRECTION OF PLAY
  ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
            arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))
  
  plt.savefig('assets/buildUp' + club + '.png', dpi=300)

  return app.get_asset_url('buildUp' + club + '.png')

################################################################################################################################################

def defensiveCover(club, data, player=None):

        color = clubColors.get(club)

        # Plotting the pitch
        fig, ax = plt.subplots(figsize=(6,4))

        pitch = Pitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.35, bottom=0.9, width=0.08, height=0.08)

        # TITLE
        fig_text(s =  club,
                x = 0.52, y = 0.96,
                color='#181818', ha='center' ,fontsize=14);

        # TITLE
        fig_text(s =  'Defensive cover',
                x = 0.515, y = 0.92,
                color='#181818', ha='center', alpha=0.8, fontsize=5);

        if data == 'WyScout':
                defensiveCover_list = defensiveCoverList(df, data)

                defensiveCover_list = defensiveCover_list.loc[defensiveCover_list['team.name'] == club].reset_index(drop=True)

                pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#e8e8e8', '#3d0000', '#ff0000'], N=10)

                path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                        path_effects.Normal()]

                pitch.scatter(defensiveCover_list['location.x'], defensiveCover_list['location.y'], ax=ax, edgecolor='white', facecolor='black', s=50, zorder=3)

                defensiveCover_list['location.x'] = defensiveCover_list['location.x'].astype(float)
                defensiveCover_list['location.y'] = defensiveCover_list['location.y'].astype(float)

                bs = pitch.bin_statistic_positional(defensiveCover_list['location.x'], defensiveCover_list['location.y'],  statistic='count', positional='full', normalize=True)
                
                pitch.heatmap_positional(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.6)

                pitch.label_heatmap(bs, color='#E8E8E8', fontsize=12,
                                        ax=ax, ha='center', va='center',
                                        str_format='{:.0%}', path_effects=path_eff)

        elif data == 'WhoScored':
                defensiveCover_list = defensiveCoverList(df, data)

                if player == None:
                  defensiveCover_list = defensiveCover_list.loc[defensiveCover_list['team'] == club].reset_index(drop=True)

                elif player != None:
                  defensiveCover_list = defensiveCover_list.loc[defensiveCover_list['name'] == player].reset_index(drop=True)

                pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#e8e8e8', color[0]], N=10)

                path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                        path_effects.Normal()]

                pitch.scatter(defensiveCover_list['x'], defensiveCover_list['y'], ax=ax, edgecolor='#181818', facecolor='#ff0000', s=15, zorder=3)

                bs = pitch.bin_statistic_positional(defensiveCover_list['x'], defensiveCover_list['y'],  statistic='count', positional='full', normalize=True)
                
                pitch.heatmap_positional(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.6)

                pitch.label_heatmap(bs, color='#E8E8E8', fontsize=8,
                                        ax=ax, ha='center', va='center',
                                        str_format='{:.0%}', path_effects=path_eff)
                
        plt.savefig('assets/defensiveCover' + club + '.png', dpi=300)
        
        return app.get_asset_url('defensiveCover' + club + '.png')

################################################################################################################################################

def dashboardDeffensive(club, matchDay, playerName, data):
        
        if data == 'WyScout':
                color = ['#041ca3', '#181818']

                fig = plt.figure(figsize=(8, 6), dpi = 300)
                grid = gridspec(6, 6)

                a1 = fig.add_subplot(grid[0:5, 0:2])
                a2 = fig.add_subplot(grid[0:5, 2:4])
                a3 = fig.add_subplot(grid[0:5, 4:9])

                #################################################################################################################################################

                #Params for the text inside the <> this is a function to highlight text
                highlight_textprops =\
                [{"color": color[0],"fontweight": 'bold'},
                {"color": color[0],"fontweight": 'bold'}]

                # Club Logo
                fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.08, bottom=0.98, width=0.2, height=0.1)

                fig.set_facecolor('#E8E8E8')

                fig_text(s =f'<{playerName}>' + "<'s>" + ' ' + 'performance',
                        x = 0.41, y = 1.07, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center' ,fontsize=14);
                
                if matchDay != 'All Season':
                        fig_text(s = 'MatchDay:' + ' ' + str(matchDay) + ' ' + '| Season 2022 | @menesesp20',
                                x = 0.33, y = 1.015 , color='#181818', fontweight='bold', ha='center' ,fontsize=5);

                if matchDay == 'All Season':
                        fig_text(s = 'Season 2022 | @menesesp20',
                                x = 0.40, y = 0.98 , color='#181818', fontweight='bold', ha='center' ,fontsize=5);

                fig_text(s = 'Territory Plot',
                        x = 0.25, y = 0.91 , color='#181818', fontweight='bold', ha='center' ,fontsize=7);

                fig_text(s = 'Pass Plot',
                        x = 0.513, y = 0.91, color='#181818', fontweight='bold', ha='center' ,fontsize=7);

                fig_text(s = 'Defensive Actions Plot',
                        x = 0.78, y = 0.91, color='#181818', fontweight='bold', ha='center' ,fontsize=7);

                #################################################################################################################################################
                # 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE

                df1 = df[(df['player.name'] == playerName) & (df['pass.accurate'] == True)]

                pitch = Pitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=a1)

                #################################################################################################################################################

                pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#E8E8E8', color[0]], N=10)

                bs = pitch.bin_statistic(df1['location.x'], df1['location.y'], bins=(10, 6))

                convex = df1[(np.abs(stats.zscore(df1[['location.x','location.y']])) < 1).all(axis=1)]

                pitch.heatmap(bs, edgecolors='#E8E8E8', ax=a1, cmap=pearl_earring_cmap)

                pitch.scatter(df1['location.x'], df1['location.y'], ax=a1, edgecolor='#181818', facecolor='black', alpha=0.3)

                hull = pitch.convexhull(convex['location.x'], convex['location.y'])

                pitch.polygon(hull, ax=a1, edgecolor='#181818', facecolor='#181818', alpha=0.4, linestyle='--', linewidth=1)

                pitch.scatter(x=convex['location.x'].mean(), y=convex['location.y'].mean(), ax=a1, c='#E8E8E8', edgecolor=color[0], s=300, zorder=2)


                #################################################################################################################################################
                # 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGUR

                player = df.loc[(df['player.name'] == playerName)]

                keyPass = player.loc[player['type.secondary'].apply(lambda x: 'key_pass' in x)]

                Pass = player.loc[(player['pass.accurate'] != 'nan')]

                sucess = player.loc[(player['pass.accurate'] != 'nan') & (player['pass.accurate'] == True)]

                unsucess = player.loc[(player['pass.accurate'] != 'nan') & (player['pass.accurate'] == False)]

                #Progressive = Pass.loc[Pass['type.secondary'].apply(lambda x: 'progressive_pass' in x)]

                Pass_percentage = round((len(sucess) / len(Pass)) * 100, 2)

                #################################################################################################################################################
                pitch = Pitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=a2)

                fig.set_facecolor('#E8E8E8')

                #Criação das setas que simbolizam os passes realizados bem sucedidos
                pitch.arrows(sucess['location.x'], sucess['location.y'], sucess['pass.endLocation.x'], sucess['pass.endLocation.y'], color='#181818', ax=a2,
                        width=1, headwidth=1, headlength=1, label='Passes' + ':' + ' ' + f'{len(Pass)}' + ' ' + '(' + f'{Pass_percentage}' + '%' + ' ' + 'Completion rate' + ')' )
                
                #Criação das setas que simbolizam os passes realizados bem sucedidos
                pitch.arrows(unsucess['location.x'], unsucess['location.y'], unsucess['pass.endLocation.x'], unsucess['pass.endLocation.y'], color='#181818', alpha=0.4, ax=a2,
                        width=1, headwidth=1, headlength=1, label='Passes unsuccessful' + ':' + ' '  + f'{len(unsucess)}')

                #Criação das setas que simbolizam os passes realizados falhados
                #pitch.arrows(Progressive['location.x'], Progressive['location.y'], Progressive['pass.endLocation.x'], Progressive['pass.endLocation.y'], color='#00bbf9', ax=a2,
                #        width=2, headwidth=5, headlength=5, label='Progressive passes' + ':' + ' ' + f'{len(Progressive)}')

                #Criação das setas que simbolizam os passes realizados falhados
                pitch.arrows(keyPass['location.x'], keyPass['location.y'], keyPass['pass.endLocation.x'], keyPass['pass.endLocation.y'], color='#ffba08', ax=a2,
                        width=1, headwidth=0.1, headlength=0.1, label='Key passes' + ':' + ' ' + f'{len(keyPass)}')
                
                pitch.scatter(keyPass['pass.endLocation.x'], keyPass['pass.endLocation.y'], s = 80, marker='*', color='#ffba08', ax=a2)

                #################################################################################################################################################

                #Criação da legenda ffba08
                l = a2.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7)
                #Ciclo FOR para atribuir a color legend
                for text in l.get_texts():
                        text.set_color("#181818")

                #################################################################################################################################################
                # 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE


                df3 = df.loc[(df['location.x'] <= 55) & (df['player.name'] == playerName)]
                

                # Tackle
                tackle = df3.loc[df3['type.secondary'].apply(lambda x: 'sliding_tackle' in x)]

                # Pressures

                pressure = df3.loc[df3['type.secondary'].apply(lambda x: 'counterpressing_recovery' in x)]

                # Interception
                interception = df3.loc[df3['type.primary'] == 'interception']

                # Aerial
                aerial = df3.loc[df3['type.secondary'].apply(lambda x: 'aerial_duel' in x)]

                # Clearance
                clearance = df3.loc[(df3['type.primary'] == 'clearance')]

                # Ball Recovery
                ballRecovery = df3.loc[df3['type.secondary'].apply(lambda x: 'recovery' in x)]
                # Plotting the pitch

                pitch = Pitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=a3)

                fig.set_facecolor('#E8E8E8')

                dfConvex = df3.loc[(df3['type.secondary'].apply(lambda x: 'sliding_tackle' in x)) | (df3['type.secondary'].apply(lambda x: 'counterpressing_recovery' in x)) |
                                (df3['type.primary'] == 'interception') | (df3['type.secondary'].apply(lambda x: 'aerial_duel' in x)) | (df3['type.primary'] == 'clearance') |
                                (df3['type.secondary'].apply(lambda x: 'recovery' in x))].reset_index(drop=True)

                convex = dfConvex.loc[(np.abs(stats.zscore(dfConvex[['location.x','location.y']])) < 1).all(axis=1)]

                hull = pitch.convexhull(convex['location.x'], convex['location.y'])

                pitch.polygon(hull, ax=a3, edgecolor='#181818', facecolor='#181818', alpha=0.3, linestyle='--', linewidth=1)

                pitch.scatter(tackle['location.x'], tackle['location.y'], ax=a3, marker='s', color='#fac404', edgecolor='#fac404', linestyle='--', s=80, label='Tackle', zorder=2)

                pitch.scatter(ballRecovery['location.x'], ballRecovery['location.y'], ax=a3, marker='8', edgecolor='#fac404', facecolor='none', hatch='//////', linestyle='--', s=80, label='Ball Recovery', zorder=2)

                pitch.scatter(aerial['location.x'], aerial['location.y'], ax=a3, marker='^', color='#fac404', edgecolor='#fac404', linestyle='--', s=80, label='Aerial', zorder=2)
                
                pitch.scatter(interception['location.x'], interception['location.y'], ax=a3, marker='P', color='#fac404', edgecolor='#fac404',  linestyle='--', s=80, label='Interception', zorder=2)

                pitch.scatter(clearance['location.x'], clearance['location.y'], ax=a3, marker='*', color='#fac404', edgecolor='#fac404', linestyle='--', s=100, label='Clearance', zorder=2)

                pitch.scatter(pressure['location.x'], pressure['location.y'], ax=a3, marker='.', color='#fac404', edgecolor='#fac404', linestyle='--', s=100, label='Pressure', zorder=2)


                #Criação da legenda
                l = a3.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7)
                #Ciclo FOR para atribuir a color legend
                for text in l.get_texts():
                        text.set_color("#181818")
        
        elif data == 'WhoScored':
                color = clubColors.get(club)

                fig = plt.figure(figsize=(8, 6), dpi = 300)
                grid = plt.GridSpec(6, 6)

                a1 = fig.add_subplot(grid[0:5, 0:2])
                a2 = fig.add_subplot(grid[0:5, 2:4])
                a3 = fig.add_subplot(grid[0:5, 4:9])

                #################################################################################################################################################

                #Params for the text inside the <> this is a function to highlight text
                highlight_textprops =\
                [{"color": color[0],"fontweight": 'bold'},
                {"color": color[0],"fontweight": 'bold'}]

                # Club Logo
                fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.05, bottom=0.85, width=0.16, height=0.1)

                fig.set_facecolor('#E8E8E8')

                fig_text(s =f'<{playerName}>' + "<'s>" + ' ' + 'performance',
                        x = 0.385, y = 0.93, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center' ,fontsize=16);
                
                if matchDay != 'All Season':
                        fig_text(s = 'World Cup Catar 2022 | @menesesp20',
                                x = 0.33, y = 0.89 , color='#181818', fontweight='bold', ha='center' ,fontsize=7);

                if matchDay == 'All Season':
                        fig_text(s = 'Season 22-23 | @menesesp20',
                                x = 0.3, y = 0.89 , color='#181818', fontweight='bold', ha='center' ,fontsize=7);

                fig_text(s = 'Territory Plot',
                        x = 0.25, y = 0.83 , color='#181818', fontweight='bold', ha='center' ,fontsize=7);

                fig_text(s = 'Pass Plot',
                        x = 0.513, y = 0.83, color='#181818', fontweight='bold', ha='center' ,fontsize=7);

                fig_text(s = 'Defensive Actions Plot',
                        x = 0.78, y = 0.83, color='#181818', fontweight='bold', ha='center' ,fontsize=7);

                #################################################################################################################################################
                # 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE

                if matchDay == 'All Season':
                        df1 = df[(df['name'] == playerName) & (df['outcomeTypedisplayName'] == 'Successful')]
                else:
                        df1 = df[(df['name'] == playerName) & (df['outcomeTypedisplayName'] == 'Successful') & (df.Match_ID == matchDay)]

                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=a1)

                #################################################################################################################################################

                pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#E8E8E8', color[0]], N=10)

                bs = pitch.bin_statistic(df1['x'], df1['y'], bins=(10, 6))

                convex = df1[(np.abs(stats.zscore(df1[['x','y']])) < 1).all(axis=1)]

                pitch.heatmap(bs, edgecolors='#E8E8E8', ax=a1, cmap=pearl_earring_cmap)

                pitch.scatter(df1['x'], df1['y'], s=30, ax=a1, edgecolor='#181818', facecolor='black', alpha=0.3)

                hull = pitch.convexhull(convex['x'], convex['y'])

                pitch.polygon(hull, ax=a1, edgecolor='#181818', facecolor='#181818', alpha=0.4, linestyle='--', linewidth=1)

                pitch.scatter(x=convex['x'].mean(), y=convex['y'].mean(), ax=a1, c='#E8E8E8', edgecolor=color[0], s=150, zorder=4)


                #################################################################################################################################################
                # 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGUR

                #mundialG['beginning'] = np.sqrt(np.square(100 - mundialG['x']) + np.square(100 - mundialG['y']))
                #mundialG['end'] = np.sqrt(np.square(100 - mundialG['endX']) + np.square(100 - mundialG['endY']))

                #mundialG['progressive'] = [(mundialG['end'][x]) / (mundialG['beginning'][x]) < .75 for x in range(len(mundialG.beginning))]

                if matchDay != 'All Season':
                        player = df.loc[(df['name'] == playerName) & (df.Match_ID == matchDay)]
                else:
                        player = df.loc[(df['name'] == playerName)]
                        
                keyPass = player.loc[player['qualifiers'].apply(lambda x: 'KeyPass' in x)]

                Pass = player.loc[(player['typedisplayName'] == 'Pass')]

                sucess = Pass.loc[(Pass['outcomeTypedisplayName'] == 'Successful')]

                unsucess = Pass.loc[(Pass['outcomeTypedisplayName'] == 'Unsuccessful')]
                
                #Progressive = Pass.loc[Pass['progressive'] == True]

                Pass_percentage = round((len(sucess) / len(Pass)) * 100, 2)

                #################################################################################################################################################
                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=a2)

                fig.set_facecolor('#E8E8E8')

                #Criação das setas que simbolizam os passes realizados bem sucedidos
                pitch.arrows(sucess['x'], sucess['y'], sucess['endX'], sucess['endY'], color='#181818', ax=a2,
                        width=1, headwidth=2, headlength=2, label='Passes' + ':' + ' ' + f'{len(Pass)}' + ' ' + '(' + f'{Pass_percentage}' + '%' + ' ' + 'Completion' + ')', zorder=5)
                
                #Criação das setas que simbolizam os passes realizados bem sucedidos
                pitch.arrows(unsucess['x'], unsucess['y'], unsucess['endX'], unsucess['endY'], color='#181818', alpha=0.4, ax=a2,
                        width=1, headwidth=2, headlength=2, label='Passes unsuccessful' + ':' + ' '  + f'{len(unsucess)}', zorder=5)

                #Criação das setas que simbolizam os passes realizados falhados
                #pitch.arrows(Progressive['x'], Progressive['y'], Progressive['endX'], Progressive['endY'], color='#00bbf9', ax=a2,
                #        width=2, headwidth=5, headlength=5, label='Progressive passes' + ':' + ' ' + f'{len(Progressive)}', zorder=5)

                #Criação das setas que simbolizam os passes realizados falhados
                pitch.arrows(keyPass['x'], keyPass['y'], keyPass['endX'], keyPass['endY'], color='#ffba08', ax=a2,
                        width=1, headwidth=0.1, headlength=0.1, label='Key passes' + ':' + ' ' + f'{len(keyPass)}', zorder=5)
                
                pitch.scatter(keyPass['endX'], keyPass['endY'], s = 30, marker='*', color='#ffba08', ax=a2, zorder=5)

                #################################################################################################################################################

                #Criação da legenda ffba08
                l = a2.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7, prop=dict(size=8))
                #Ciclo FOR para atribuir a color legend
                for text in l.get_texts():
                        text.set_color("#181818")

                #################################################################################################################################################
                # 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE

                #if matchDay != 'All Season':
                #        df3 = df.loc[(df['x'] <= 55) & (df['name'] == playerName) & (df['Match_ID'] == matchDay)]
                #else:
                #        df3 = df.loc[(df['x'] <= 55) & (df['name'] == playerName)]
                
                if matchDay != 'All Season':        
                    df3 = df.loc[(df['name'] == playerName) & (df['Match_ID'] == matchDay)]
                else:
                    df3 = df.loc[(df['name'] == playerName)]
                
                # Tackle
                tackle = df3.loc[(df3['typedisplayName'] == 'Tackle') & (df3['outcomeTypedisplayName'] == 'Successful')]

                # Pressures
                #pressure = df3.loc[df3['type.secondary'].apply(lambda x: 'counterpressing_recovery' in x)]

                # Interception
                interception = df3.loc[df3['typedisplayName'] == 'Interception']

                # Aerial
                aerial = df3.loc[(df3['typedisplayName'] == 'Aerial') & (df3['outcomeTypedisplayName'] == 'Successful')]
                
                aerialUn = df3.loc[(df3['typedisplayName'] == 'Aerial') & (df3['outcomeTypedisplayName'] == 'Unsuccessful')]

                Aerial_percentage = round((len(aerial) / (len(aerial) + len(aerialUn))) * 100, 2)

                # Clearance
                clearance = df3.loc[(df3['typedisplayName'] == 'Clearance') & (df3['outcomeTypedisplayName'] == 'Successful')]

                # Ball Recovery
                ballRecovery = df3.loc[(df3['typedisplayName'] == 'BallRecovery')]
                
                # Plotting the pitch
                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=a3)

                fig.set_facecolor('#E8E8E8')

                dfConvex = df3.loc[(df3['typedisplayName'] == 'BallRecovery') | (df3['typedisplayName'] == 'Clearance') |
                                   (df3['typedisplayName'] == 'Aerial') | (df3['typedisplayName'] == 'Interception') |
                                   (df3['typedisplayName'] == 'Tackle')].reset_index(drop=True)

                convex = dfConvex.loc[(np.abs(stats.zscore(dfConvex[['x','y']])) < 1).all(axis=1)]

                hull = pitch.convexhull(convex['x'], convex['y'])

                pitch.polygon(hull, ax=a3, edgecolor='#181818', facecolor='#181818', alpha=0.3, linestyle='--', linewidth=1)

                pitch.scatter(tackle['x'], tackle['y'], ax=a3, marker='s', color='#fac404', edgecolor='#fac404', linestyle='--', s=30, label='Tackle', zorder=2)

                pitch.scatter(ballRecovery['x'], ballRecovery['y'], ax=a3, marker='8', edgecolor='#fac404', facecolor='none', hatch='//////', linestyle='--', s=30, label='Ball Recovery', zorder=2)

                pitch.scatter(aerial['x'], aerial['y'], ax=a3, marker='^', color='#fac404', edgecolor='#fac404', linestyle='--', s=30, label='Aerial ' + f'{len(aerial)}' + '/' + f'{len(aerial)}' + ' ' + '(' + f'{Aerial_percentage}' + '%' + ' ' + 'Completion' + ')', zorder=2)
                
                pitch.scatter(interception['x'], interception['y'], ax=a3, marker='P', color='#fac404', edgecolor='#fac404',  linestyle='--', s=30, label='Interception', zorder=2)

                pitch.scatter(clearance['x'], clearance['y'], ax=a3, marker='*', color='#fac404', edgecolor='#fac404', linestyle='--', s=50, label='Clearance', zorder=2)

                #pitch.scatter(pressure['x'], pressure['y'], ax=a3, marker='.', color='#fac404', edgecolor='#fac404', linestyle='--', s=200, label='Pressure', zorder=2)


                #Criação da legenda
                l = a3.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7, prop=dict(size=8))
                #Ciclo FOR para atribuir a color legend
                for text in l.get_texts():
                        text.set_color("#181818")
                        
        plt.savefig('assets/defensiveDashboard' + playerName + '.png', dpi=300)
        
        return app.get_asset_url('defensiveDashboard' + playerName + '.png')

################################################################################################################################################

def dashboardOffensive(club, playerName, matchDay, data):

        color = clubColors.get(club)

        fig = plt.figure(figsize=(6, 4), dpi = 300)
        grid = plt.GridSpec(8, 8)

        a1 = fig.add_subplot(grid[1:7, 0:2])
        a2 = fig.add_subplot(grid[1:7, 2:4])
        a3 = fig.add_subplot(grid[1:7, 4:6])
        a4 = fig.add_subplot(grid[1:7, 6:8])

        #################################################################################################################################################

        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
        [{"color": color[0],"fontweight": 'bold'},
        {"color": color[0],"fontweight": 'bold'}]

        # Club Logo
        add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.05, bottom=0.85, width=0.16, height=0.1)

        fig.set_facecolor('#E8E8E8')

        #fig_text(s='All Pases', color='#e4dst54', highlight_textprops = highlight_textprops)

        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
        [{"color": color[0],"fontweight": 'bold'},
        {"color": color[0],"fontweight": 'bold'}]


        fig_text(s =f'<{playerName}>' + "<'s>" + ' ' + 'performance',
                 x = 0.45, y = 0.92, color='#181818', highlight_textprops = highlight_textprops, fontweight='bold', ha='center', fontsize=14);
        
        if matchDay != 'All Season':
                fig_text(s = 'MatchDay' + ' ' + str(matchDay) + '| Season 2022-23 | @menesesp20',
                        x = 0.33, y = 0.88,
                        color='#181818', fontweight='bold', ha='center' ,fontsize=8, alpha=0.8);

        elif matchDay == 'All Season':
                fig_text(s ='World Cup Catar 2022 | @menesesp20',
                        x = 0.33, y = 0.88,
                        color='#181818', fontweight='bold', ha='center' ,fontsize=5, alpha=0.8);

        fig_text(s = 'Territory Plot',
                 x = 0.22, y = 0.75, color='#181818', fontweight='bold', ha='center' ,fontsize=7);

        fig_text(s = 'Pass Plot',
                 x = 0.41, y = 0.75, color='#181818', fontweight='bold', ha='center' ,fontsize=7);

        fig_text(s = 'xT Plot',
                 x = 0.61, y = 0.75, color='#181818', fontweight='bold', ha='center' ,fontsize=7);

        fig_text(s = 'Offensive Actions',
                 x = 0.81, y = 0.75, color='#181818', fontweight='bold', ha='center' ,fontsize=7);

        if data == 'WyScout':
                #################################################################################################################################################
                # 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE

                df1 = df.loc[(df['player.name'] == playerName) & (df['pass.accurate'] == True)]

                pitch = VerticalPitch(pitch_type='opta', pad_top=0.1, pad_bottom=0.5,
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=1, linewidth=3, spot_scale=0.00)

                pitch.draw(ax=a1)

                #################################################################################################################################################

                pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#E8E8E8', color[0]], N=10)
                bs = pitch.bin_statistic(df1['location.x'], df1['location.y'], bins=(12, 8))

                pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#E8E8E8', color[0]], N=10)
                bs = pitch.bin_statistic(df1['location.x'], df1['location.y'], bins=(12, 8))

                convex = df1[(np.abs(stats.zscore(df1[['location.x','location.y']])) < 1).all(axis=1)]

                pitch.heatmap(bs, edgecolors='#E8E8E8', ax=a1, cmap=pearl_earring_cmap)

                pitch.scatter(df1['location.x'], df1['location.y'], ax=a1, s=30, edgecolor='#181818', facecolor='black', alpha=0.3)

                hull = pitch.convexhull(convex['location.x'], convex['location.y'])

                pitch.polygon(hull, ax=a1, edgecolor='#181818', facecolor='#181818', alpha=0.4, linestyle='--', linewidth=2.5)

                pitch.scatter(x=convex['location.x'].mean(), y=convex['location.y'].mean(), ax=a1, c='white', edgecolor=color[0], s=100, zorder=2)


                #################################################################################################################################################
                # 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGUR

                player = df.loc[(df['player.name'] == playerName)]

                keyPass = player.loc[player['type.secondary'].apply(lambda x: 'key_pass' in x)]

                Pass = player.loc[(player['pass.accurate'] != 'nan')]

                sucess = player.loc[(player['pass.accurate'] != 'nan') & (player['pass.accurate'] == True)]

                unsucess = player.loc[(player['pass.accurate'] != 'nan') & (player['pass.accurate'] == False)]

                Progressive = Pass.loc[Pass['type.secondary'].apply(lambda x: 'progressive_pass' in x)]

                Pass_percentage = round((len(sucess) / len(Pass)) * 100, 2)

                #################################################################################################################################################
                pitch = VerticalPitch(pitch_type='opta', pad_top=0.1, pad_bottom=0.5,
                        pitch_color='#E8E8E8', line_color='#181818',
                        line_zorder=1, linewidth=3, spot_scale=0.00)

                pitch.draw(ax=a2)

                fig.set_facecolor('#E8E8E8')

                #Criação das setas que simbolizam os passes realizados bem sucedidos
                pitch.arrows(sucess['location.x'], sucess['location.y'], sucess['pass.endLocation.x'], sucess['pass.endLocation.y'], color='#181818', ax=a2,
                        width=2, headwidth=5, headlength=5, label='Passes' + ':' + ' ' + '76' + ' ' + '(' + '88' + '%' + ' ' + 'Completion rate' + ')' )
                
                #Criação das setas que simbolizam os passes realizados bem sucedidos
                pitch.arrows(unsucess['location.x'], unsucess['location.y'], unsucess['pass.endLocation.x'], unsucess['pass.endLocation.y'], color='#cad2c5', ax=a2,
                        width=2, headwidth=5, headlength=5, label='Passes unsuccessful' + ':' + ' '  + '9')

                #Criação das setas que simbolizam os passes realizados falhados
                pitch.arrows(Progressive['location.x'], Progressive['location.y'], Progressive['pass.endLocation.x'], Progressive['pass.endLocation.y'], color='#00bbf9', ax=a2,
                        width=2, headwidth=5, headlength=5, label='Progressive passes' + ':' + ' ' + f'{len(Progressive)}')

                #Criação das setas que simbolizam os passes realizados falhados
                pitch.arrows(keyPass['location.x'], keyPass['location.y'], keyPass['pass.endLocation.x'], keyPass['pass.endLocation.y'], color='#ffba08', ax=a2,
                        width=2, headwidth=0.1, headlength=0.1, label='Key passes' + ':' + ' ' + f'{len(keyPass)}')
                
                pitch.scatter(keyPass['pass.endLocation.y'], keyPass['pass.endLocation.y'], s = 100, marker='*', color='#ffba08', ax=a2)

                #################################################################################################################################################

                #Criação da legenda
                l = a2.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7)
                #Ciclo FOR para atribuir a white color na legend
                for text in l.get_texts():
                        text.set_color("#181818")

                #################################################################################################################################################
                # 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE

                xTDF = xT(events, data)

                xTheatMap = xTDF.loc[(xTDF.xT > 0) & (xTDF['player.name'] == playerName)]

                # setup pitch
                pitch = VerticalPitch(pitch_type='opta', pad_top=0.1, pad_bottom=0.5,
                        pitch_color='#E8E8E8', line_color='#181818', line_zorder=1, linewidth=3, spot_scale=0.00)

                pitch.draw(ax=a3)

                fig.set_facecolor('#E8E8E8')

                pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                        ['#E8E8E8', color[0]], N=10)

                bs = pitch.bin_statistic(xTheatMap['location.x'], xTheatMap['location.y'], bins=(12, 8))

                heatmap = pitch.heatmap(bs, edgecolors='#E8E8E8', ax=a3, cmap=pearl_earring_cmap)

                #################################################################################################################################################
                # 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE


                df4 = events.loc[(events['player.name'] == playerName)]
                
                # carry
                carries = df4.loc[df4['type.secondary'].apply(lambda x: 'carry' in x)]

                # deep_completion
                deep_completion = df4.loc[df4['type.secondary'].apply(lambda x: 'deep_completion' in x)]

                # smart_pass
                smart_pass = df4.loc[df4['type.primary'] == 'smart_pass']

                # dribble
                dribble = df4.loc[df4['type.secondary'].apply(lambda x: 'dribble' in x)]

                # Plotting the pitch
                pitch = VerticalPitch(pitch_type='opta', pad_top=0.1, pad_bottom=0.5,
                                        pitch_color='#E8E8E8', line_color='#181818',
                                        line_zorder=1, linewidth=5, spot_scale=0.005)

                pitch.draw(ax=a4)

                fig.set_facecolor('#E8E8E8')

                pitch.lines(carries['location.x'], carries['location.y'],
                                carries['carry.endLocation.x'], carries['carry.endLocation.y'],
                                lw=2, ls='dashed', label='Carry' + ':' + ' ' + f'{len(carries)}',
                                color='#ffba08', ax=a4 ,zorder=4)

                pitch.arrows(deep_completion['location.x'], deep_completion['location.y'], deep_completion['pass.endLocation.x'], deep_completion['pass.endLocation.y'],
                        color=color[0], ax=a4,
                        width=2, headwidth=5, headlength=5, label='Deep Completion' + ':' + ' ' + f'{len(deep_completion)}', zorder=4)

                pitch.arrows(smart_pass['location.x'], smart_pass['location.y'], smart_pass['pass.endLocation.x'], smart_pass['pass.endLocation.y'], color='#ffba08', ax=a4,
                        width=2,headwidth=5, headlength=5, label='Smart pass' + ':' + ' ' + f'{len(smart_pass)}', zorder=4)

                pitch.scatter(dribble['location.x'], dribble['location.y'], s = 100, marker='*', color='#ffba08', ax=a4,
                        label='Dribble' + ':' + ' ' + f'{len(dribble)}', zorder=4)


                #Criação da legenda
                l = a4.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7)
                #Ciclo FOR para atribuir a white color na legend
                for text in l.get_texts():
                        text.set_color("#181818")
                        
        elif data == 'WhoScored':

                if matchDay != 'All Season':
                        events = df.loc[df.Match_ID == matchDay].reset_index(drop=True)
                else:
                        events = df.copy()
                #################################################################################################################################################
                # 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE

                df1 = events.loc[(events['name'] == playerName) & (events['typedisplayName'] == 'Pass')]

                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=a1)

                #################################################################################################################################################

                pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#E8E8E8', color[0]], N=10)
                bs = pitch.bin_statistic(df1['x'], df1['y'], bins=(12, 8))

                convex = df1[(np.abs(stats.zscore(df1[['x','y']])) < 1).all(axis=1)]

                pitch.heatmap(bs, edgecolors='#E8E8E8', ax=a1, cmap=pearl_earring_cmap)

                pitch.scatter(df1['x'], df1['y'], ax=a1, s=15, edgecolor='#181818', facecolor='black', alpha=0.3)

                hull = pitch.convexhull(convex['x'], convex['y'])

                pitch.polygon(hull, ax=a1, edgecolor='#181818', facecolor='#181818', alpha=0.4, linestyle='--', linewidth=1)

                pitch.scatter(x=convex['x'].mean(), y=convex['y'].mean(), ax=a1, c='white', edgecolor=color[0], s=80, zorder=5)


                #################################################################################################################################################
                # 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGUR

                #df['beginning'] = np.sqrt(np.square(100 - df['x']) + np.square(100 - df['y']))
                #df['end'] = np.sqrt(np.square(100 - df['endX']) + np.square(100 - df['endY']))

                #df['progressive'] = [(df['end'][x]) / (df['beginning'][x]) < .75 for x in range(len(df.beginning))]

                player = events.loc[(events['name'] == playerName)]

                keyPass = player.loc[player['qualifiers'].apply(lambda x: 'KeyPass' in x)]

                Pass = player.loc[(player['typedisplayName'] == 'Pass')]

                sucess = Pass.loc[(Pass['outcomeTypedisplayName'] == 'Successful')]

                unsucess = Pass.loc[(Pass['outcomeTypedisplayName'] == 'Unsuccessful')]

                #Progressive = Pass.loc[Pass['progressive'] == True]

                Pass_percentage = round((len(sucess) / len(Pass)) * 100, 2)

                #################################################################################################################################################
                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=a2)

                fig.set_facecolor('#E8E8E8')

                #Criação das setas que simbolizam os passes realizados bem sucedidos
                pitch.arrows(sucess['x'], sucess['y'], sucess['endX'], sucess['endY'], color='#181818', ax=a2,
                        width=1, headwidth=1, headlength=1, label='Passes' + ':' + ' ' + '76' + ' ' + '(' + '88' + '%' + ' ' + 'Completion' + ')' )
                
                #Criação das setas que simbolizam os passes realizados bem sucedidos
                pitch.arrows(unsucess['x'], unsucess['y'], unsucess['endX'], unsucess['endY'], color='#cad2c5', ax=a2,
                        width=1, headwidth=1, headlength=1, label='Passes unsuccessful' + ':' + ' '  + '9')

                #Criação das setas que simbolizam os passes realizados falhados
                #pitch.arrows(Progressive['x'], Progressive['y'], Progressive['endX'], Progressive['endY'], color='#00bbf9', ax=a2,
                #        width=2, headwidth=5, headlength=5, label='Progressive passes' + ':' + ' ' + f'{len(Progressive)}')

                #Criação das setas que simbolizam os passes realizados falhados
                pitch.arrows(keyPass['x'], keyPass['y'], keyPass['endX'], keyPass['endY'], color='#ffba08', ax=a2,
                        width=1, headwidth=0.1, headlength=0.1, label='Key passes' + ':' + ' ' + f'{len(keyPass)}')
                
                pitch.scatter(keyPass['endX'], keyPass['endY'], s = 15, marker='*', color='#ffba08', ax=a2)

                #################################################################################################################################################

                #Criação da legenda
                l = a2.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7, prop=dict(size=5))
                #Ciclo FOR para atribuir a white color na legend
                for text in l.get_texts():
                        text.set_color("#181818")

                #################################################################################################################################################
                # 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE

                xTDF = xT(events, data)

                xTheatMap = xTDF.loc[(xTDF.xT > 0) & (xTDF['name'] == playerName)]

                # setup pitch
                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=a3)

                fig.set_facecolor('#E8E8E8')

                pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                        ['#E8E8E8', color[0]], N=10)

                bs = pitch.bin_statistic(xTheatMap['x'], xTheatMap['y'], bins=(12, 8))

                heatmap = pitch.heatmap(bs, edgecolors='#E8E8E8', ax=a3, cmap=pearl_earring_cmap)

                #################################################################################################################################################
                # 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE

                df4 = events.loc[(events['name'] == playerName)].reset_index(drop=True)
                
                # carry
                #matchId = df.Match_ID.unique()
                #dataAppend = []
                #for game in matchId:
                #        data = carry(events, club, game, carrydf=None, progressive=None)
                #        dataAppend.append(data)

                #carries = pd.concat(dataAppend)
                #carries.reset_index(drop=True, inplace=True)
                
                #carries = carries.loc[(carries.typedisplayName == 'Carry') & (carries.name == playerName)].reset_index(drop=True)

                #carriesProgressive = carry(events, club, matchDay, carrydf=None, progressive=None)
                #carriesProgressive = carriesProgressive.loc[(carriesProgressive.progressiveCarry == True) & (carries.name == playerName)].reset_index(drop=True)

                # deep_completion
                #deep_completion = df4.loc[df4['type.secondary'].apply(lambda x: 'deep_completion' in x)]

                # smart_pass
                smart_pass = df4.loc[df4['qualifiers'].apply(lambda x: 'KeyPass' in x)].reset_index(drop=True)

                # dribble
                dribble = df4.loc[df4['typedisplayName'] == 'TakeOn'].reset_index(drop=True)

                # Plotting the pitch
                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=a4)

                fig.set_facecolor('#E8E8E8')

                #pitch.lines(carries['x'], carries['y'],
                #        carries['endX'], carries['endY'],
                #        lw=2, ls='dashed', label='Carry' + ':' + ' ' + f'{len(carries)}',
                #        color='#ffba08', ax=a4 ,zorder=4)

                #pitch.lines(carriesProgressive['x'], carriesProgressive['y'],
                #        carriesProgressive['endX'], carriesProgressive['endY'],
                #        lw=2, ls='dashed', label='Progressive Carry' + ':' + ' ' + f'{len(carriesProgressive)}',
                #        color='#ea04dc', ax=a4 ,zorder=4)

                #pitch.arrows(deep_completion['x'], deep_completion['y'],
                #             deep_completion['endX'], deep_completion['endY'],
                #             color=color[0], ax=a4, width=2, headwidth=5, headlength=5,
                #             label='Deep Completion' + ':' + ' ' + f'{len(deep_completion)}', zorder=4)

                pitch.arrows(smart_pass['x'], smart_pass['y'],
                        smart_pass['endX'], smart_pass['endY'],
                        color='#ffba08', ax=a4, width=1,headwidth=1, headlength=1,
                        label='Key Pass' + ':' + ' ' + f'{len(smart_pass)}', zorder=4)

                pitch.scatter(dribble['x'], dribble['y'],
                        s = 50, marker='*', color='#ffba08', ax=a4,
                        label='Dribble' + ':' + ' ' + f'{len(dribble)}', zorder=4)


                #Criação da legenda
                l = a4.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7, prop=dict(size=5))
                #Ciclo FOR para atribuir a white color na legend
                for text in l.get_texts():
                        text.set_color("#181818")

        plt.savefig('assets/offensiveDashboard' + playerName + '.png', dpi=300)
        
        return app.get_asset_url('offensiveDashboard' + playerName + '.png')

################################################################################################################################################

def plotZone14Passes(x):
    
        pitch = VerticalPitch(pitch_type='opta', half=True,
                pitch_color='#E8E8E8', line_color='#181818',
                line_zorder=1, linewidth=0.5, spot_scale=0.005)

        pitch.draw(ax=x)

        ZONE14 = patches.Rectangle([68, 35], width=15.05, height=35, linewidth = 1, linestyle='-',
                                        edgecolor='#800000', facecolor='#800000', alpha=0.8)

        # ZONE 14 VERTICAL PITCH
        ZONE14 = patches.Rectangle([33, 63], width=33, height=20, linewidth = 0.8, linestyle='-',
                                edgecolor='#181818', facecolor='#800000', alpha=0.8)

        x.add_patch(ZONE14)

################################################################################################################################################

def horizontalBar(data, col_player, col, x=None):

  if x==None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    #Set color background outside the graph
    fig.set_facecolor('#E8E8E8')

    #Set color background inside the graph
    ax.set_facecolor('#E8E8E8')

    for i in range(len(data)):
      plt.barh(data[col_player], data[col], fill=True, color='#800000', edgecolor='#181818', linewidth=1)
    
    ax.set_ylabel(col_player, size=11, color='#181818', fontweight='bold', labelpad=50)

    ax.set_xlabel(col, size=11, color='#181818', fontweight='bold', labelpad=12)

    #Atribuição da cor e tamanho das tick labels, the left=False retires the ticks
    ax.tick_params(axis='x', colors='#181818', labelsize=8)
    ax.tick_params(axis='y', colors='#181818', labelsize=8, left = False)

    #Setg the color of the line in the spines and retire the spines from the top and right sides
    ax.spines['bottom'].set_color('#181818')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#181818')
    ax.spines['right'].set_visible(False)

    #Bold the labels
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    
    fig.text(0.03, 0.05,'Made by Pedro Meneses/@menesesp20.', color='#181818', size=5)
    
  elif x != None:
    
    for i in range(len(data)):
      x.barh(data[col_player], data[col], fill=True, color='#800000', edgecolor='#181818', linewidth=1)
    
    x.set_ylabel(col_player, size=11, color='#181818', fontweight='bold', labelpad=16)

    x.set_xlabel(col, size=11, color='#181818', fontweight='bold', labelpad=8)

    #Atribuição da cor e tamanho das tick labels, the left=False retires the ticks
    x.tick_params(axis='x', colors='#181818', labelsize=8)
    x.tick_params(axis='y', colors='#181818', labelsize=8, left = False)

    #Setg the color of the line in the spines and retire the spines from the top and right sides
    x.spines['bottom'].set_color('#181818')
    x.spines['top'].set_visible(False)
    x.spines['left'].set_color('#181818')
    x.spines['right'].set_visible(False)

    #Bold the labels
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"   

################################################################################################################################################

def plotDasboardZone14(team, data):
    
    if data == 'WyScout':
        zone14 = df.loc[(df['type.primary'] == 'pass') & (df['location.x'] >= 70) & (df['location.x'] <= 83) & (df['location.y'] >= 36) & (df['location.y'] <= 63.5)].reset_index(drop=True)

        #Criação da lista de jogadores
        Players = zone14['player.name'].unique()

        zone14Passes = []

        #Ciclo For de atribuição dos valores a cada jogador
        for player in Players:
            zone14Passes.append(zone14.loc[zone14['player.name'] == player, 'player.name'].count())
            
        data = {
            'Players' : Players,
            'Zone14' : zone14Passes
            }

        zone14 = pd.DataFrame(data)
    
    elif data == 'WhoScored':
        zone14 = df.loc[(df['team'] == team) & (df['typedisplayName'] == 'Pass') & (df['x'] >= 70) & (df['x'] <= 83) & (df['y'] >= 36) & (df['y'] <= 63.5)].reset_index(drop=True)

        #Criação da lista de jogadores
        Players = zone14['name'].unique()

        zone14Passes = []

        #Ciclo For de atribuição dos valores a cada jogador
        for player in Players:
            zone14Passes.append(zone14.loc[zone14['name'] == player, 'name'].count())
            
        data = {
            'Players' : Players,
            'Zone14' : zone14Passes
            }

        zone14 = pd.DataFrame(data)

    fig = plt.figure(figsize=(10, 6), dpi = 300)
    grid = plt.GridSpec(8, 8)

    a1 = fig.add_subplot(grid[2:6, 2:4])
    a2 = fig.add_subplot(grid[3:5, 4:7])
        
    fig.set_facecolor('#E8E8E8')
    
    a1.set_facecolor('#E8E8E8')
    
    #################################################################################################################################################

    # Club Logo
    add_image(image='Images/Clubs/' + 'Mundial' + '/' + team + '.png', fig=fig, left=0.12, bottom=0.825, width=0.2, height=0.1)

    fig_text(s = 'The master at finding space in Zone 14',
                x = 0.5, y = 0.88,
                color='#181818', fontweight='bold',
                ha='center', va='center',
                fontsize=18)

    fig_text(s = 'World Cup Catar 2022 | @Menesesp20',
                x = 0.37, y = 0.84,
                color='#181818', fontweight='bold',
                ha='center', va='center',
                fontsize=8, alpha=0.5)
    
    horizontalBar(zone14.sort_values('Zone14', ascending=True), 'Players', 'Zone14', a1)
    
    plotZone14Passes(a2)
    
    plt.savefig('assets/dashboard14' + team + '.png', dpi=300)
    
    return app.get_asset_url('dashboard14' + team + '.png')

################################################################################################################################################

def defensiveLine(team, data):

    if data == 'WyScout':
        # Defensive Actions
        defensiveActions = df.loc[(df['team.name'] == team) &
                                  ((df['type.secondary'].apply(lambda x: 'sliding_tackle' in x)) |
                                   (df['type.secondary'].apply(lambda x: 'counterpressing_recovery' in x)) |
                                   (df['type.secondary'].apply(lambda x: 'interception' in x)) |
                                   (df['type.secondary'].apply(lambda x: 'aerial_duel' in x)) |
                                   (df['type.secondary'].apply(lambda x: 'clearance' in x)) |
                                   (df['type.secondary'].apply(lambda x: 'recovery' in x)))].reset_index(drop=True)

        # Plotting the pitch
        fig, ax = plt.subplots(figsize=(6,4))

        pitch = Pitch(pitch_type='opta',
                      pitch_color='#E8E8E8', line_color='#181818',
                      line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        fig_text(s = team + "'s defensive line",
                    x = 0.53, y = 0.92, fontweight='bold',
                    ha='center',fontsize=12, color='#181818');

        #Linha média do eixo x
        plt.axhline(defensiveActions['location.x'].mean(), c='#ff0000', linestyle='--', LineWidth=2)

        #Color a span inside the graph to define the peak age of a player
        plt.axhspan(defensiveActions['location.x'].mean(), -50, facecolor='#ff0000', alpha=0.4)
        
        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + team + '.png', fig=fig, left=0.345, bottom=0.885, width=0.08, height=0.05)
    
    elif data == 'WhoScored':
        # Defensive Actions
        defensiveActions = df.loc[(df['team'] == team) & ((df['typedisplayName'] == 'BallRecovery') |
                                                (df['typedisplayName'] == 'Tackle') |
                                                (df['typedisplayName'] == 'Interception') |
                                                (df['typedisplayName'] == 'Aerial') |
                                                (df['typedisplayName'] == 'Clearance'))].reset_index(drop=True)

        # Plotting the pitch
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        pitch = VerticalPitch(pitch_type='opta',
                            pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=1, linewidth=2, spot_scale=0.005)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        fig_text(s = team + "'s defensive line",
                    x = 0.53, y = 0.94, fontweight='bold',
                    ha='center',fontsize=12, color='#181818');

        fig_text(s = str(round(defensiveActions['x'].mean(), 2)) + 'm',
                 x = 0.408, y = 0.52, color='#181818', fontweight='bold', ha='center', alpha=0.8, fontsize=5);

        #Linha média do eixo x
        plt.axhline(defensiveActions['x'].mean(), c='#ff0000', linestyle='--')

        #Color a span inside the graph to define the peak age of a player
        plt.axhspan(defensiveActions['x'].mean(), -50, facecolor='#ff0000', alpha=0.4)
        
        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + team + '.png', fig=fig, left=0.345, bottom=0.85, width=0.05, height=0.05)
        
        plt.savefig('assets/defensiveLine' + team + '.png', dpi=300)
        
        return app.get_asset_url('defensiveLine' + team + '.png')        

################################################################################################################################################

def xT_Flow(club, gameDay, data):

    color = clubColors.get(club)

    df_XT = df.loc[(df['outcomeTypedisplayName'] == 'Successful') & (df['Match_ID'] == gameDay)].reset_index(drop=True)

    xTDF = xT(df_XT, data)

    dfxT = dataFramexTFlow(xTDF, club, data)

    dfxT['xTH'] = dfxT['home_xT'].rolling(window=5).mean()

    dfxT['xTH'] = round(dfxT['xTH'], 2)

    dfxT['xTA'] = dfxT['away_xT'].rolling(window=5).mean()

    dfxT['xTA'] = round(dfxT['xTA'], 2)

    #Drop rows with NaN values
    dfxT = dfxT.dropna(axis=0, subset=['xTH', 'xTA'])

    fig, ax = plt.subplots(figsize=(20,12))

    #Set color background outside the graph
    fig.set_facecolor('#e8e8e8')

    #Set color background inside the graph
    ax.set_facecolor('#e8e8e8')

    home = df_XT.home_Team.unique()
    homeName = home[0]
    color = clubColors.get(homeName)

    away = df_XT.away_Team.unique()
    awayName = away[0]
    color2 = clubColors.get(awayName)

    dfxT['xTH'] = dfxT['home_xT'].rolling(window = 5, min_periods = 0).mean()

    dfxT['xTA'] = dfxT['away_xT'].rolling(window = 5, min_periods = 0).mean()

    ax.fill_between(dfxT.Minutes, dfxT['xTH'], 0,
                    where=(dfxT['xTH'] > dfxT['xTA']),
                    interpolate=True, color=color[0], edgecolor='white', lw=3)

    #ax.fill(df.Minutes, df['xTH'], "r", df.Minutes, df['xTA'], "b")

    ax.fill_between(dfxT.Minutes, -abs(dfxT['xTA']), 0,
                    where=(dfxT['xTA'] > dfxT['xTH']),
                    interpolate=True, color=color2[0], edgecolor='white', lw=3)

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
          [{"color": color[0], "fontweight": 'bold'}
          ]

    #Title
    Title = fig_text(s = f'<{club}>' + ' ' + 'xT Flow',
                     x = 0.48, y = 0.97, highlight_textprops = highlight_textprops ,fontweight='bold', ha='center', fontsize=50 ,color='#181818');

    fig_text(s = 'World Cup Catar 2022 | xT values based on Karun Singhs model | @menesesp20',
             x = 0.5, y = 0.92, fontweight='bold',
             ha='center',fontsize=16, color='#181818', alpha=0.4);

    # Half Time Line
    halfTime = 45

    ax.axvline(halfTime, color='#181818', ls='--', lw=1)

    diferencialLine = 0
    ax.axhline(diferencialLine, color='#181818', ls='-', lw=1.5)

    fig_text(s = 'HALF TIME',
             x = 0.525, y = 0.85, fontweight='bold',
             ha='center',fontsize=5, color='#181818');


    #Atribuição da cor e tamanho das tick labels, the left=False retires the ticks
    ax.tick_params(axis='x', colors='#181818', labelsize=14)
    ax.tick_params(axis='y', colors='#181818', labelsize=14, left = False)
    
    #Setg the color of the line in the spines and retire the spines from the top and right sides
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #Bold the labels
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"

    # Club Logo
    fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.1, bottom=0.855, width=0.1, height=0.15)
    
    plt.savefig('assets/xTFlow' + club + '.png', dpi=300)
    
    return app.get_asset_url('xTFlow' + club + '.png')   

################################################################################################################################################

def touch_Flow(club):

    color = clubColors.get(club)

    df = dataFrame_touchFlow(df, club)

    df['touchHome'] = df['home_Touches'].rolling(window = 5, min_periods = 0).mean()

    df['touchHome'] = round(df['touchHome'], 2)

    df['touchAway'] = df['away_Touches'].rolling(window = 5, min_periods = 0).mean()

    df['touchAway'] = round(df['touchAway'], 2)

    #Drop rows with NaN values
    df = df.dropna(axis=0, subset=['touchHome', 'away_Touches'])

    fig, ax = plt.subplots(figsize=(20,12))

    #Set color background outside the graph
    fig.set_facecolor('#E8E8E8')

    #Set color background inside the graph
    ax.set_facecolor('#E8E8E8')

    ax.fill_between(df.Minutes, df['touchHome'], 0,
                    where=(df['touchHome'] > df['touchAway']),
                    interpolate=True, color=color[0], edgecolor='#181818', lw=3)

    ax.fill_between(df.Minutes, -abs(df['touchAway']), 0,
                    where=(df['touchAway'] > df['touchHome']),
                    interpolate=True, color='#ff0000', edgecolor='#181818', lw=3)

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
          [{"color": color[0],"fontweight": 'bold'},
            {"color": color[0],"fontweight": 'bold'},
            {"color": "#ff0000","fontweight": 'bold'},
            {"color": "#ff0000","fontweight": 'bold'}
          ]

    home = df.Home.unique()
    homeName = home[0]
  
    away = df.Away.unique()
    awayName = away[0]

    Goal_Home = df.Goal_Home.unique()
    Goal_Home = Goal_Home[0]
  
    Goal_Away = df.Goal_Away.unique()
    Goal_Away = Goal_Away[0]

    #Title
    Title = fig_text(s = f'<{homeName}>' + ' ' + f'<{Goal_Home}>' + ' ' + '-' + ' ' + f'<{Goal_Away}>' + ' ' + f'<{awayName}>',
                     x = 0.438, y = 0.93, highlight_textprops = highlight_textprops,
                     fontweight='bold', ha='center', fontsize=14, color='#181818');

    fig_text(s = 'World Cup Catar 2022 | Passes Final 3rd flow graph | @menesesp20',
             x = 0.43, y = 0.89,
             fontweight='bold', ha='center', fontsize=5, color='#181818', alpha=0.4);

    # Half Time Line
    halfTime = 45

    ax.axvline(halfTime, color='#181818', ls='--', lw=1)

    diferencialLine = 0

    ax.axhline(diferencialLine, color='#181818', ls='-', lw=1.5)

    fig_text(s = 'HALF TIME',
             x = 0.525, y = 0.85,
             fontweight='bold', ha='center', fontsize=5, color='#181818');

    #Atribuição da cor e tamanho das tick labels, the left=False retires the ticks
    ax.tick_params(axis='x', colors='#181818', labelsize=5)
    ax.tick_params(axis='y', colors='#181818', labelsize=5, left = False)

    #Setg the color of the line in the spines and retire the spines from the top and right sides
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #Bold the labels
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"

    # Club Logo
    fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.08, bottom=0.925, width=0.2, height=0.1)
    
    plt.savefig('assets/TouchFlow' + club + '.png', dpi=300)
    
    return app.get_asset_url('TouchFlow' + club + '.png')

################################################################################################################################################

def GoalKick(club, data):

        if data == 'WyScout':
                #################################################################################################################################################
                
                goalKick = cluster_Event(df, club, 'goal_kick', 3, data)

                #################################################################################################################################################

                # Plotting the pitch

                fig, ax = plt.subplots(figsize=(6,4))

                pitch = Pitch(pitch_type='opta',
                            pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=ax)

                fig.set_facecolor('#E8E8E8')

                #################################################################################################################################################

                # Title of our plot

                fig.suptitle('How do they come out playing?', fontsize=50, color='#181818',
                        fontweight = "bold", x=0.53, y=0.95)

                fig_text(s = "GoalKick | Season 21-22 | Made by: @menesesp20",
                        x = 0.5, y = 0.9,
                        color='#181818', fontweight='bold', ha='center' ,fontsize=16);

                #################################################################################################################################################

                # Key Passes Cluster
                for x in range(len(goalKick['cluster'])):
                
                        # First
                        if goalKick['cluster'][x] == 0:
                                pitch.arrows(xstart=goalKick['location.x'][x], ystart=goalKick['location.y'][x],
                                        xend=goalKick['pass.endLocation.x'][x], yend=goalKick['pass.endLocation.y'][x],
                                        color='#ea04dc', alpha=0.8,
                                        lw=3, zorder=2,
                                        ax=ax)
                                
                        # Second
                        if goalKick['cluster'][x] == 2:
                                pitch.arrows(xstart=goalKick['location.x'][x], ystart=goalKick['location.y'][x],
                                        xend=goalKick['pass.endLocation.x'][x], yend=goalKick['pass.endLocation.y'][x],
                                        color='#2d92df', alpha=0.8,
                                        lw=3, zorder=2,
                                        ax=ax)
                        
                        # Third
                        if goalKick['cluster'][x] == 1:
                                pitch.arrows(xstart=goalKick['location.x'][x], ystart=goalKick['location.y'][x],
                                        xend=goalKick['pass.endLocation.x'][x], yend=goalKick['pass.endLocation.y'][x],
                                        color='#fb8c04', alpha=0.8,
                                        lw=3, zorder=2,
                                        ax=ax)

                #################################################################################################################################################

                fig_text(s = 'Most frequent zone',
                        x = 0.8, y = 0.79,
                        color='#ea04dc', fontweight='bold', ha='center' ,fontsize=12);

                fig_text(s = 'Second most frequent zone',
                        x = 0.8, y = 0.76,
                        color='#2d92df', fontweight='bold', ha='center' ,fontsize=12);

                fig_text(s = 'Third most frequent zone',
                        x = 0.8, y = 0.73,
                        color='#fb8c04', fontweight='bold', ha='center' ,fontsize=12);

                # Club Logo
                fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.1, bottom=0.865, width=0.2, height=0.1)

                fig_text(s = 'Attacking Direction',
                                x = 0.5, y = 0.17,
                                color='#181818', fontweight='bold',
                                ha='center', va='center',
                                fontsize=14)

                # ARROW DIRECTION OF PLAY
                ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                        arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))
        
        if data == 'WhoScored':
                #################################################################################################################################################
        
                goalKick = cluster_Event(df, club, 'GoalKick', 3, data)

                #################################################################################################################################################

                # Plotting the pitch

                fig, ax = plt.subplots(figsize=(6,4))

                pitch = Pitch(pitch_type='opta',
                            pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=ax)

                fig.set_facecolor('#E8E8E8')

                #################################################################################################################################################

                # Title of our plot

                fig.suptitle('How do they come out playing?', fontsize=14, color='#181818',
                        fontweight = "bold", x=0.53, y=0.93)

                fig_text(s = "GoalKick | World Cup Catar 2022 | @menesesp20",
                        x = 0.5, y = 0.89,
                        color='#181818', fontweight='bold', ha='center', fontsize=5);

                #################################################################################################################################################

                # Key Passes Cluster
                for x in range(len(goalKick['cluster'])):
                
                        # First
                        if goalKick['cluster'][x] == 0:
                                pitch.arrows(xstart=goalKick['x'][x], ystart=goalKick['y'][x],
                                        xend=goalKick['endX'][x], yend=goalKick['endY'][x],
                                        color='#ea04dc', alpha=0.8,
                                        lw=1, zorder=2,
                                        ax=ax)
                                
                        # Second
                        if goalKick['cluster'][x] == 2:
                                pitch.arrows(xstart=goalKick['x'][x], ystart=goalKick['y'][x],
                                        xend=goalKick['endX'][x], yend=goalKick['endY'][x],
                                        color='#2d92df', alpha=0.8,
                                        lw=1, zorder=2,
                                        ax=ax)
                        
                        # Third
                        if goalKick['cluster'][x] == 1:
                                pitch.arrows(xstart=goalKick['x'][x], ystart=goalKick['y'][x],
                                        xend=goalKick['endX'][x], yend=goalKick['endY'][x],
                                        color='#fb8c04', alpha=0.8,
                                        lw=1, zorder=2,
                                        ax=ax)

                #################################################################################################################################################

                fig_text(s = 'Most frequent zone',
                        x = 0.8, y = 0.79,
                        color='#ea04dc', fontweight='bold', ha='center' ,fontsize=5);

                fig_text(s = 'Second most frequent zone',
                        x = 0.8, y = 0.76,
                        color='#2d92df', fontweight='bold', ha='center' ,fontsize=5);

                fig_text(s = 'Third most frequent zone',
                        x = 0.8, y = 0.73,
                        color='#fb8c04', fontweight='bold', ha='center' ,fontsize=5);

                # Club Logo
                fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.1, bottom=0.85, width=0.05, height=0.1)

                fig_text(s = 'Attacking Direction',
                                x = 0.5, y = 0.17,
                                color='#181818', fontweight='bold',
                                ha='center', va='center',
                                fontsize=8)

                # ARROW DIRECTION OF PLAY
                ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                        arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))
        
        plt.savefig('assets/GoalKick' + club + '.png', dpi=300)

        return app.get_asset_url('GoalKick' + club + '.png')

################################################################################################################################################

def counterPressMap(team, data, player=None):

    # Plotting the pitch
    fig, ax = plt.subplots(figsize=(6,4))

    pitch = VerticalPitch(pitch_type='opta',
                pitch_color='#E8E8E8', line_color='#181818',
                line_zorder=3, linewidth=0.5, spot_scale=0.00)

    pitch.draw(ax=ax)

    fig.set_facecolor('#E8E8E8')

    fig_text(s = team + ' counter press',
                x = 0.53, y = 0.93, fontweight='bold',
                ha='center',fontsize=14, color='#181818');

    fig_text(s = 'World Cup Catar 2022 | @Menesesp20',
                x = 0.53, y = 0.89, fontweight='bold',
                ha='center',fontsize=8, color='#181818', alpha=0.4);

    # Club Logo
    fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + team + '.png', fig=fig, left=0.32, bottom=0.85, width=0.05, height=0.07)
    
    if data == 'WyScout':
        # Counter Press DataFrame
        counterDF = counterPress(df, team, data)

        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                            ['#e8e8e8', '#3d0000', '#ff0000'], N=10)

        path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                    path_effects.Normal()]

        counterDF['location.x'] = counterDF['location.x'].astype(float)
        counterDF['location.y'] = counterDF['location.y'].astype(float)

        bs = pitch.bin_statistic_positional(counterDF['location.x'], counterDF['location.y'],  statistic='count', positional='full', normalize=True)
        
        pitch.heatmap_positional(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.6)

        pitch.label_heatmap(bs, color='#E8E8E8', fontsize=8,
                                    ax=ax, ha='center', va='center',
                                    str_format='{:.0%}', path_effects=path_eff)
        
    elif data == 'WhoScored':
        
        # Counter Press DataFrame
        dataCP = counterPress(team, data)
        
        if player == None:
            dataCP = dataCP.loc[dataCP.typedisplayName == 'BallRecovery'].reset_index(drop=True)
        else:
            dataCP = dataCP.loc[(dataCP.typedisplayName == 'BallRecovery') & (dataCP.name == player)].reset_index(drop=True)

        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                            ['#e8e8e8', '#3d0000', '#ff0000'], N=10)

        path_eff = [path_effects.Stroke(linewidth=1, foreground='black'),
                    path_effects.Normal()]

        bs = pitch.bin_statistic_positional(dataCP['x'], dataCP['y'],  statistic='count', positional='full', normalize=True)
        
        pitch.heatmap_positional(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.6)

        pitch.label_heatmap(bs, color='#E8E8E8', fontsize=5,
                                    ax=ax, ha='center', va='center',
                                    str_format='{:.0%}', path_effects=path_eff)

        plt.savefig('assets/counterPress' + team + '.png', dpi=300)

        return app.get_asset_url('counterPress' + team + '.png')

################################################################################################################################################

def through_passMap(gameID, club, data, playerName=None):

        color = ['#FF0000', '#181818']

        if data == 'WyScout':
                if playerName == None:
                        player_Df = df.loc[df['team.name'] == club].reset_index(drop=True)
                else:
                        player_Df = df.loc[df['player.name'] == playerName].reset_index(drop=True)

        elif data == 'WhoScored':
                if playerName == None:
                        player_Df = df.loc[(df['team'] == club) & (df['Match_ID'] == gameID)].reset_index(drop=True)
                else:
                        player_Df = df.loc[(df['name'] == playerName) & (df['Match_ID'] == gameID)].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(6,4))

        pitch = Pitch(pitch_type='opta',
                    pitch_color='#E8E8E8', line_color='#181818',
                    line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        ###############################################################################################################################################################
        ###############################################################################################################################################################

        if data == 'WyScout':
                through_pass = df.loc[df['type.secondary'].apply(lambda x: 'through_pass' in x)].reset_index(drop=True)

                through_passSucc = through_pass.loc[through_pass['pass.accurate'] == True].reset_index(drop=True)

                through_passUnsucc = through_pass.loc[through_pass['pass.accurate'] == False].reset_index(drop=True)

                through_passKP = through_pass.loc[through_pass['type.secondary'].apply(lambda x: 'key_pass' in x)].reset_index(drop=True)

                through_passAst = through_pass.loc[through_pass['type.secondary'].apply(lambda x: 'assist' in x)].reset_index(drop=True)

                ###############################################################################################################################################################
                ###############################################################################################################################################################

                # Plot Through Passes Successful
                pitch.lines(through_passSucc['location.x'], through_passSucc['location.y'], through_passSucc['pass.endLocation.x'], through_passSucc['pass.endLocation.y'],
                        lw=5, color=color[0], comet=True,
                        label='Through Passes Successful', ax=ax)

                pitch.scatter(through_passSucc['pass.endLocation.x'], through_passSucc['pass.endLocation.y'], s=100,
                        marker='o', edgecolors=color[0], c=color[0], zorder=3, ax=ax)

                # Plot Through Passes Unsuccessful
                pitch.lines(through_passUnsucc['location.x'], through_passUnsucc['location.y'], through_passUnsucc['pass.endLocation.x'], through_passUnsucc['pass.endLocation.y'],
                        lw=5, color='#ff0000', comet=True,
                        label='Through Passes Unsuccessful', ax=ax)

                pitch.scatter(through_passUnsucc['pass.endLocation.x'], through_passUnsucc['pass.endLocation.y'], s=100,
                        marker='o', edgecolors='#ff0000', c='#ff0000', zorder=3, ax=ax)

                for i in range(len(through_pass)):
                        plt.text(through_pass['location.x'].values[i] + 0.7, through_pass['location.y'].values[i] + 0.7, through_pass['player.name'].values[i], color=color[0], zorder=5)

                for i in range(len(through_passSucc)):        
                        plt.text(through_passSucc['pass.endLocation.x'].values[i] + 0.7, through_passSucc['pass.endLocation.y'].values[i] + 0.7, through_passSucc['pass.recipient.name'].values[i], color=color[0], zorder=5)
                
                for i in range(len(through_passKP)):
                        plt.text(through_passKP['pass.endLocation.x'].values[i] + 0.7, through_passKP['pass.endLocation.y'].values[i] + 0.7, through_passKP['pass.recipient.name'].values[i], color=color[0], zorder=5)

                ###############################################################################################################################################################
                ###############################################################################################################################################################
                
                # Plot Key Passes
                pitch.lines(through_passKP['location.x'], through_passKP['location.y'], through_passKP['pass.endLocation.x'], through_passKP['pass.endLocation.y'],
                        lw=5, color='#ffba08', comet=True,
                        label='Key Passes', ax=ax)

                # Plot Key Passes
                pitch.scatter(through_passKP['pass.endLocation.x'], through_passKP['pass.endLocation.y'], s=100,
                        marker='o', edgecolors='#ffba08', c='#ffba08', zorder=3, ax=ax)

                ###############################################################################################################################################################
                ###############################################################################################################################################################
                
                # Plot Key Passes
                pitch.lines(through_passAst['location.x'], through_passAst['location.y'], through_passAst['pass.endLocation.x'], through_passAst['pass.endLocation.y'],
                        lw=5, color='#fb8c04', comet=True,
                        label='Assist', ax=ax)

                # Plot Key Passes
                pitch.scatter(through_passAst['pass.endLocation.x'], through_passAst['pass.endLocation.y'], s=100,
                        marker='o', edgecolors='#fb8c04', c='#fb8c04', zorder=3, ax=ax)

        elif data == 'WhoScored':
                
                #identify the passer and then the recipient, who'll be the playerId of the next action
                player_Df['passer'] = player_Df['name']

                player_Df['recipient'] = player_Df['passer'].shift(+1)
                
                through_pass = player_Df.loc[player_Df['qualifiers'].apply(lambda x: 'Throughball' in x)].reset_index(drop=True)

                through_passSucc = through_pass.loc[through_pass['outcomeTypedisplayName'] == 'Successful'].reset_index(drop=True)

                through_passUnsucc = through_pass.loc[through_pass['outcomeTypedisplayName'] == 'Unsuccessful'].reset_index(drop=True)

                through_passKP = through_pass.loc[through_pass['qualifiers'].apply(lambda x: 'KeyPass' in x)].reset_index(drop=True)

                through_passAst = through_pass.loc[through_pass['qualifiers'].apply(lambda x: 'IntentionalGoalAssist' in x)].reset_index(drop=True)

                ###############################################################################################################################################################
                ###############################################################################################################################################################

                # Plot Through Passes Successful
                pitch.lines(through_passSucc['x'], through_passSucc['y'], through_passSucc['endX'], through_passSucc['endY'],
                        lw=5, color='#08d311', comet=True,
                        label='Through Passes Successful', ax=ax)

                pitch.scatter(through_passSucc['endX'], through_passSucc['endY'], s=50,
                        marker='o', edgecolors='#08d311', c="#08d311", zorder=3, ax=ax)

                # Plot Through Passes Unsuccessful
                pitch.lines(through_passUnsucc['x'], through_passUnsucc['y'], through_passUnsucc['endX'], through_passUnsucc['endY'],
                        lw=5, color='#ff0000', comet=True,
                        label='Through Passes Unsuccessful', ax=ax)

                pitch.scatter(through_passUnsucc['endX'], through_passUnsucc['endY'], s=50,
                        marker='o', edgecolors='#ff0000', c='#ff0000', zorder=3, ax=ax)

                for i in range(len(through_pass)):
                        plt.text(through_pass['x'].values[i] + 0.7, through_pass['y'].values[i] + 0.7, through_pass['name'].values[i], color=color[0], zorder=5)

                for i in range(len(through_passSucc)):        
                        plt.text(through_passSucc['endX'].values[i] + 0.7, through_passSucc['endY'].values[i] + 0.7, through_passSucc['recipient'].values[i], color=color[0], zorder=5)
                
                for i in range(len(through_passKP)):
                        plt.text(through_passKP['endX'].values[i] + 0.7, through_passKP['endY'].values[i] + 0.7, through_passKP['recipient'].values[i], color=color[0], zorder=5)

                ###############################################################################################################################################################
                ###############################################################################################################################################################
                
                # Plot Key Passes
                pitch.lines(through_passKP['x'], through_passKP['y'], through_passKP['endX'], through_passKP['endY'],
                        lw=5, color='#ffba08', comet=True,
                        label='Key Passes', ax=ax)

                # Plot Key Passes
                pitch.scatter(through_passKP['endX'], through_passKP['endY'], s=50,
                        marker='o', edgecolors='#ffba08', c='#ffba08', zorder=3, ax=ax)

                ###############################################################################################################################################################
                ###############################################################################################################################################################
                
                # Plot Key Passes
                pitch.lines(through_passAst['x'], through_passAst['y'], through_passAst['endX'], through_passAst['endY'],
                        lw=5, color='#fb8c04', comet=True,
                        label='Assist', ax=ax)

                # Plot Key Passes
                pitch.scatter(through_passAst['endX'], through_passAst['endY'], s=50,
                        marker='o', edgecolors='#fb8c04', c='#fb8c04', zorder=3, ax=ax)

        ###############################################################################################################################################################
        ###############################################################################################################################################################
        
        #Criação da legenda
        l = ax.legend(bbox_to_anchor=(0.02, 1), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7)
        #Ciclo FOR para atribuir a color legend
        for text in l.get_texts():
                text.set_color("#181818")

        ###############################################################################################################################################################
        ###############################################################################################################################################################

        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
        [{"color": color[0], "fontweight": 'bold'}]

        if (playerName == None) & (gameID != 'All Season'):
                fig_text(s =f'<{club}>' + ' ' + 'Throughballs',
                        x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center', va='center', fontsize=14);
                
                fig_text(s ='MatchDay:' + str(gameID) + ' ' +  '| Season 21-22 | @menesesp20',
                        x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=5, alpha=0.7);

        elif (playerName == None) & (gameID == 'All Season'):
                fig_text(s =f'<{club}>' + ' ' + 'Throughballs',
                        x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center', va='center', fontsize=14);
                
                fig_text(s ='All Season' + ' ' +  '| World Cup Catar 2022 | @menesesp20',
                        x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=5, alpha=0.7);

        if (playerName != None) & (gameID != 'All Season'):
                fig_text(s =f'<{playerName}>' + ' ' + 'Throughballs',
                        x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center', va='center', fontsize=14);
                
                fig_text(s ='MatchDay:' + str(gameID) + ' ' +  '| Season 21-22 | @menesesp20',
                        x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=5, alpha=0.7);

        elif (playerName != None) & (gameID == 'All Season'):
                fig_text(s =f'<{club}>' + ' ' + 'Throughballs',
                        x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                        color='#181818', fontweight='bold', ha='center', va='center', fontsize=14);
                
                fig_text(s ='All Season' + ' ' +  '| World Cup Catar 2022 | @menesesp20',
                        x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=5, alpha=0.7);
        
        ###############################################################################################################################################################
        ###############################################################################################################################################################
        

        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.08, bottom=0.87, width=0.2, height=0.08)

        fig_text(s = 'Attacking Direction',
                        x = 0.5, y = 0.17,
                        color='#181818', fontweight='bold',
                        ha='center', va='center',
                        fontsize=8)

        # ARROW DIRECTION OF PLAY
        ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))

        plt.savefig('assets/lineBreak' + club + '.png', dpi=300)

        return app.get_asset_url('lineBreak' + club + '.png')

################################################################################################################################################

def ShotMap(team, playerName=None):

  dfGoal = df.loc[(df['shot.isGoal'] == True) | (df['shot.isGoal'] == False)].reset_index(drop=True)

  home = dfGoal['team.name'].unique()
  home = home[0]
  color = ['#041ca3']

  away = dfGoal['opponentTeam.name'].unique()
  away = away[0]

  color = ['#041ca3', '#181818']

  fig, ax = plt.subplots(figsize=(6,4))

  pitch = Pitch(pitch_type='opta',
            pitch_color='#E8E8E8', line_color='#181818',
            line_zorder=3, linewidth=0.5, spot_scale=0.00)

  pitch.draw(ax=ax)

  fig.set_facecolor('#E8E8E8')

  for i in range(len(dfGoal)):
    if (dfGoal['shot.isGoal'].values[i] == True) & (dfGoal['team.name'].values[i] == team):
    
      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(dfGoal['location.x'].values[i], dfGoal['location.y'].values[i],
                    color=color[0], marker='h', edgecolors='#ff0000', lw=2, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500) + 100,
                    zorder=3)

      plt.text(dfGoal['location.x'].values[i] + 1.2, dfGoal['location.y'].values[i] + 1, dfGoal['player.name'].values[i], zorder=4)

    elif (dfGoal['shot.isGoal'].values[i] == True) & (dfGoal['team.name'].values[i] == team) & (dfGoal['shot.xg'].values[i] <= 0.05):
    
      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(dfGoal['location.x'].values[i], dfGoal['location.y'].values[i],
                    color='#fb8c04', marker='h', edgecolors='#181818', lw=2, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500) + 100,
                    zorder=3)

      plt.text(dfGoal['location.x'].values[i] + 1.2, dfGoal['location.y'].values[i] + 1, dfGoal['player.name'].values[i], zorder=4)

    elif (dfGoal['shot.isGoal'].values[i] == True) & (dfGoal['team.name'].values[i] == team) & (dfGoal['shot.xg'].values[i] >= 0.7):
    
      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(dfGoal['location.x'].values[i], dfGoal['location.y'].values[i],
                    color='#ea04dc', marker='h', edgecolors='#181818', lw=2, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500) + 100,
                    zorder=3)

      plt.text(dfGoal['location.x'].values[i] + 1.2, dfGoal['location.y'].values[i] + 1, dfGoal['player.name'].values[i], zorder=4)

    elif (dfGoal['shot.isGoal'].values[i] == False) & (dfGoal['team.name'].values[i] == team):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(dfGoal['location.x'].values[i], dfGoal['location.y'].values[i],
                    color=color[0], alpha=0.7, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)

    elif (dfGoal['shot.isGoal'].values[i] == True) & (dfGoal['team.name'].values[i] == team) & (df['shot.bodyPart'].values[i] == 'head_or_other'):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color=color[0], marker='8', edgecolors='#ff0000', lw=2, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)

      plt.text(100 - dfGoal['location.x'].values[i] + 1.2, 100 - dfGoal['location.y'].values[i] + 1, dfGoal['player.name'].values[i], zorder=4)

    elif (dfGoal['shot.isGoal'].values[i] == False) & (dfGoal['team.name'].values[i] == team) & (df['shot.bodyPart'].values[i] == 'head_or_other'):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color=color[0], marker='8', ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)

    elif (dfGoal['shot.isGoal'].values[i] == True) & (dfGoal['team.name'].values[i] == team) & ((dfGoal['type.secondary'].apply(lambda x: 'shot_after_corner' in x).values[i] | (dfGoal['type.secondary'].apply(lambda x: 'shot_after_free_kick' in x)).values[i])):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color=color[0], marker='s', edgecolors='#ff0000', lw=1, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)

      plt.text(100 - dfGoal['location.x'].values[i] + 1.2, 100 - dfGoal['location.y'].values[i] + 1, dfGoal['player.name'].values[i], zorder=4)

    elif (dfGoal['shot.isGoal'].values[i] == False) & (dfGoal['team.name'].values[i] == team) & ((dfGoal['type.secondary'].apply(lambda x: 'shot_after_corner' in x).values[i]) | (dfGoal['type.secondary'].apply(lambda x: 'shot_after_free_kick' in x)).values[i]):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color=color[0], marker='s', ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)
    #######################################################################################################################################3

    elif (dfGoal['shot.isGoal'].values[i] == False) & (dfGoal['team.name'].values[i] != team):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color=color[0], alpha=0.7, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)

    elif (dfGoal['shot.isGoal'].values[i] == True) & (dfGoal['team.name'].values[i] != team):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color=color[0], marker='h', edgecolors='#ff0000', lw=2, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)

      plt.text(100 - dfGoal['location.x'].values[i] + 1.2, 100 - dfGoal['location.y'].values[i] + 1, dfGoal['player.name'].values[i], zorder=4)

    elif (dfGoal['shot.isGoal'].values[i] == True) & (dfGoal['team.name'].values[i] != team) & (dfGoal['shot.xg'].values[i] <= 0.05):
    
      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color='#fb8c04', marker='h', edgecolors='#181818', lw=2, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500) + 100,
                    zorder=3)

      plt.text(100 - dfGoal['location.x'].values[i] + 1.2, 100 - dfGoal['location.y'].values[i] + 1, dfGoal['player.name'].values[i], zorder=4)

    elif (dfGoal['shot.isGoal'].values[i] == True) & (dfGoal['team.name'].values[i] != team) & (dfGoal['shot.xg'].values[i] >= 0.7):
    
      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color='#ea04dc', marker='h', edgecolors='#181818', lw=2, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500) + 100,
                    zorder=3)

      plt.text(100 - dfGoal['location.x'].values[i] + 1.2, 100 - dfGoal['location.y'].values[i] + 1, dfGoal['player.name'].values[i], zorder=4)

    elif (dfGoal['shot.isGoal'].values[i] == True) & (dfGoal['team.name'].values[i] != team) & (df['shot.bodyPart'].values[i] == 'head_or_other'):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color=color[0], marker='8', edgecolors='#ff0000', lw=2, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)

      plt.text(100 - dfGoal['location.x'].values[i] + 1.2, 100 - dfGoal['location.y'].values[i] + 1, dfGoal['player.name'].values[i], zorder=4)

    elif (dfGoal['shot.isGoal'].values[i] == False) & (dfGoal['team.name'].values[i] != team) & (df['shot.bodyPart'].values[i] == 'head_or_other'):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color=color[0], marker='8', ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)

    elif (dfGoal['shot.isGoal'].values[i] == True) & (dfGoal['team.name'].values[i] != team) & ((dfGoal['type.secondary'].apply(lambda x: 'shot_after_corner' in x)).values[i] | (dfGoal['type.secondary'].values[i].apply(lambda x: 'shot_after_free_kick' in x)).values[i]):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color=color[0], marker='s', edgecolors='#ff0000', lw=1, ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)

      plt.text(100 - dfGoal['location.x'].values[i] + 1.2, 100 - dfGoal['location.y'].values[i] + 1, dfGoal['player.name'].values[i], zorder=4)

    elif (dfGoal['shot.isGoal'].values[i] == False) & (dfGoal['team.name'].values[i] != team) & ((dfGoal['type.secondary'].apply(lambda x: 'shot_after_corner' in x)).values[i] | (dfGoal['type.secondary'].apply(lambda x: 'shot_after_free_kick' in x)).values[i]):

      #Criação das setas que simbolizam os passes realizados bem sucedidos
      pitch.scatter(100 - dfGoal['location.x'].values[i], 100 - dfGoal['location.y'].values[i],
                    color=color[0], marker='s', ax=ax, s=(dfGoal['shot.xg'].values[i] * 1500),
                    zorder=3)
                    
     ##################################################################################################################################################################
     ##################################################################################################################################################################
  
    ax.scatter(2, -2.2, color='#e8e8e8', marker='s', lw=2, edgecolors='#181818', s=200,
              zorder=3)

    ax.text(3.2, -2.7, 'Set Piece', color='#181818', size=12,
              zorder=3)

    ax.scatter(9.8, -2.2, color='#e8e8e8', marker='8', lw=2, edgecolors='#181818', s=200,
              zorder=3)

    ax.text(11, -2.7, 'Header', color='#181818', size=12,
              zorder=3)

    ax.scatter(16.3, -2.2, color='#e8e8e8', marker='h', lw=2, edgecolors='#ff0000', s=200,
              zorder=3)

    ax.text(17.7, -2.7, 'Goal', color='#181818', size=12,
              zorder=3)

    ax.scatter(22, -2.2, color='#e8e8e8', lw=2, edgecolors='#181818', s=200,
              zorder=3)

    ax.text(23.5, -2.7, 'Shot', color='#181818', size=12,
              zorder=3)

    ax.scatter(28, -2.2, color='#fb8c04', marker='h', lw=2, edgecolors='#181818', s=200,
              zorder=3)

    ax.text(29.5, -2.7, 'Low xG', color='#181818', size=12,
              zorder=3)

    ax.scatter(35, -2.2, color='#ea04dc', marker='h', lw=2, edgecolors='#181818', s=200,
              zorder=3)

    ax.text(36.3, -2.7, 'High xG', color='#181818', size=12,
              zorder=3)

    #Params for the text inside the <> this is a function to highlight text
  highlight_textprops =\
    [{"color": color[0],"fontweight": 'bold'},
    {"color": '#ff0000',"fontweight": 'bold'}]
    
  fig_text(s =f'<{home}>' + ' ' + 'vs' + ' ' + f'<{away}>',
             x = 0.53, y = 0.93,
             ha='center', va='center',
             highlight_textprops = highlight_textprops, 
             color='#181818', fontweight='bold',
             fontsize=45);

  fig_text(s = 'Shot Map',
            x = 0.505, y = 0.9,
            color='#181818', fontweight='bold', ha='center', va='center',fontsize=23);

  fig_text(s =  'league' + ' ' + '|' + ' ' + 'MatchDay:' + ' ' + str(1) + ' ' + '| Season 21-22 | @menesesp20',
            x = 0.5, y = 0.87,
            color='#181818', fontweight='bold', ha='center', va='center',fontsize=18);

  # Club Logo
  fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + team + '.png', fig=fig, left=0.12, bottom=0.885, width=0.2, height=0.08)
  
  plt.savefig('assets/shotMap' + team + '.png', dpi=300)

  return app.get_asset_url('shotMap' + team + '.png')

################################################################################################################################################

def halfspaces_Zone14(club):

    Game = df.loc[(df['name'] == club) & (df['typedisplayName'] == 'Pass')]

    fig, ax = plt.subplots(figsize=(6,4))

    pitch = Pitch(pitch_type='opta',
            pitch_color='#E8E8E8', line_color='#181818',
            line_zorder=3, linewidth=0.5, spot_scale=0.00)

    pitch.draw(ax=ax)

    fig.set_facecolor('#E8E8E8')

    ###################################################################################################################################

    fig.suptitle(club, fontsize=14, color='#181818', fontweight = "bold", y=0.93)

    Title = fig_text(s = 'Half Spaces Zone 14 passes | World Cup Catar 2022 | @menesesp20',
                     x = 0.51, y = 0.89, color='#181818', ha='center',
                     fontweight = "bold", fontsize=5);

    ###################################################################################################################################

    ZONE14 = patches.Rectangle([20.8, 68], width=58, height=15, linewidth = 1, linestyle='-',
                            edgecolor='#181818', facecolor='#ff0000', alpha=0.5, zorder=1 )

    HalfSpaceLeft = patches.Rectangle([67, 67.8], width=20, height=78, linewidth = 1, linestyle='-',
                            edgecolor='#181818', facecolor='#2894e5', alpha=0.5, zorder=1 )

    HalfSpaceRight = patches.Rectangle([13, 67.8], width=20, height=78, linewidth = 1, linestyle='-',
                            edgecolor='#181818', facecolor='#2894e5', alpha=0.5, zorder=1 )

    ###################################################################################################################################

    # HALF SPACE LEFT

    halfspaceleft = Game[(Game['endY'] <= 83) & (Game['endY'] >= 65) &
                                  (Game['endX'] >= 78)]

    pitch.arrows(xstart=halfspaceleft['x'], ystart=halfspaceleft['y'],
                                        xend=halfspaceleft['endX'], yend=halfspaceleft['endY'],
                                        color='#2894e5', alpha=0.8,
                                        lw=3, zorder=3,
                                        ax=ax)

    ###################################################################################################################################

    # ZONE14

    zone14 = Game[(Game['endX'] <= 83) & (Game['endX'] >= 75) &
                          (Game['endY'] <= 66) & (Game['endY'] >= 35)]

    pitch.arrows(xstart=zone14['x'], ystart=zone14['y'],
                                        xend=zone14['endX'], yend=zone14['endY'],
                                        color='#ff0000', alpha=0.8,
                                        lw=3, zorder=3,
                                        ax=ax)

    ###################################################################################################################################

    # HALF SPACE RIGHT

    halfspaceright = Game[(Game['endY'] >= 17) & (Game['endY'] <= 33) &
                          (Game['endX'] >= 78)]

    pitch.arrows(xstart=halfspaceright['x'], ystart=halfspaceright['y'],
                                        xend=halfspaceright['endX'], yend=halfspaceright['endYy'],
                                        color='#2894e5', alpha=0.8,
                                        lw=3, zorder=3,
                                        ax=ax)

    ###################################################################################################################################

    ax.add_patch(ZONE14)
    ax.add_patch(HalfSpaceLeft)
    ax.add_patch(HalfSpaceRight)

    ###################################################################################################################################

    # Club Logo
    fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.24, bottom=0.85, width=0.05, height=0.1)

    plt.savefig('assets/halfSpace' + club + '.png', dpi=300)

    return app.get_asset_url('halfSpace' + club + '.png')

################################################################################################################################################

def finalThird(club, matchDay, data):

        if data == 'WyScout':
                if matchDay != 'All Season':
                        # DATAFRAME WITH ALL PASSES IN THE FINAL THIRD
                        final3rd = df.loc[(df['pass.accurate'] == True) & (df['team.name'] == club) &
                                        (df['location.x'] >= 55) & (df['Match_ID'] == matchDay)][['team.name', 'player.name', 'location.x', 'location.y',
                                                                                                'pass.endLocation.y', 'pass.endLocation.x',
                                                                                                'type.primary', 'type.secondary', 'pass.accurate']]

                elif matchDay == 'All Season':
                        # DATAFRAME WITH ALL PASSES IN THE FINAL THIRD
                        final3rd = df.loc[(df['pass.accurate'] == True) & (df['team.name'] == club) &
                                (df['location.x'] >= 55)][['team.name', 'player.name', 'location.x', 'location.y', 'pass.endLocation.y', 'pass.endLocation.x',
                                                        'type.primary', 'type.secondary', 'pass.accurate']]

                # DATAFRAME WITH ALL PASSES IN THE LEFT FINAL THIRD
                #67 LEFT, RIGHT 33, MID BEETWEN THEM
                leftfinal3rd = final3rd[(final3rd['location.y'] >= 67)]

                # PERCENTAGE OF ATTACKS IN THE LEFT SIDE
                leftfinal3rdTotal = round((len(leftfinal3rd) / len(final3rd)) * 100 ,1)

                # DATAFRAME WITH ALL PASSES IN THE CENTER FINAL THIRD
                centerfinal3rd = final3rd[(final3rd['location.y'] < 67) & (final3rd['location.y'] > 33)]

                # PERCENTAGE OF ATTACKS IN THE CENTER SIDE
                centerfinal3rdTotal = round((len(centerfinal3rd) / len(final3rd)) * 100 ,1)

                # DATAFRAME WITH ALL PASSES IN THE RIGHT FINAL THIRD
                rightfinal3rd = final3rd[(final3rd['location.y'] <= 33)]

                # PERCENTAGE OF ATTACKS IN THE RIGHT SIDE
                rightfinal3rdTotal = round((len(rightfinal3rd) / len(final3rd)) * 100 ,1)

                #################################################################################################################################################

                final3rd_Cluster = cluster_Event(df, club, 'key_pass', 4, data)
                
                #################################################################################################################################################
                df = df.loc[df['pass.accurate'] == True].reset_index(drop=True)

                xTDF = xT(df, data)

                DFSides = sides(xTDF, data, club)

                xT_Sides = dataFrame_xTFlow(DFSides)

        if data == 'WhoScored':
                if matchDay != 'All Season':
                        # DATAFRAME WITH ALL PASSES IN THE FINAL THIRD
                        final3rd = df.loc[(df['typedisplayName'] == 'Pass') & (df['team'] == club) &
                                          (df['x'] >= 55) & (df['Match_ID'] == matchDay)][['team', 'name', 'x', 'y', 'endX', 'endY', 'typedisplayName', 'outcomeTypedisplayName']]

                elif matchDay == 'All Season':
                        # DATAFRAME WITH ALL PASSES IN THE FINAL THIRD
                        final3rd = df.loc[(df['qualifiers'].str.contains('KeyPass') == True) &
                                          (df['team'] == club) & (df['x'] >= 55)][['team', 'name', 'x', 'y', 'endX', 'endY', 'typedisplayName', 'outcomeTypedisplayName']]

                # DATAFRAME WITH ALL PASSES IN THE LEFT FINAL THIRD
                #67 LEFT, RIGHT 33, MID BEETWEN THEM
                leftfinal3rd = final3rd[(final3rd['y'] >= 67)]

                # PERCENTAGE OF ATTACKS IN THE LEFT SIDE
                leftfinal3rdTotal = round((len(leftfinal3rd) / len(final3rd)) * 100 ,1)

                # DATAFRAME WITH ALL PASSES IN THE CENTER FINAL THIRD
                centerfinal3rd = final3rd[(final3rd['y'] < 67) & (final3rd['y'] > 33)]

                # PERCENTAGE OF ATTACKS IN THE CENTER SIDE
                centerfinal3rdTotal = round((len(centerfinal3rd) / len(final3rd)) * 100 ,1)

                # DATAFRAME WITH ALL PASSES IN THE RIGHT FINAL THIRD
                rightfinal3rd = final3rd[(final3rd['y'] <= 33)]

                # PERCENTAGE OF ATTACKS IN THE RIGHT SIDE
                rightfinal3rdTotal = round((len(rightfinal3rd) / len(final3rd)) * 100 ,1)

                #################################################################################################################################################

                final3rd_Cluster = cluster_Event(club, 'KeyPass', 4, data)

                final3rd_Cluster0 = final3rd_Cluster.loc[final3rd_Cluster.cluster == 0]
                final3rd_Cluster1 = final3rd_Cluster.loc[final3rd_Cluster.cluster == 1]
                final3rd_Cluster2 = final3rd_Cluster.loc[final3rd_Cluster.cluster == 2]
                
                x_mean0 = final3rd_Cluster0.x.mean()
                y_mean0 = final3rd_Cluster0.y.mean()

                x_end_mean0 = final3rd_Cluster0.endX.mean()
                y_end__mean0 = final3rd_Cluster0.endY.mean()

                x_mean1 = final3rd_Cluster1.x.mean()
                y_mean1 = final3rd_Cluster1.y.mean()

                x_end_mean1 = final3rd_Cluster1.endX.mean()
                y_end__mean1 = final3rd_Cluster1.endY.mean()

                x_mean2 = final3rd_Cluster2.x.mean()
                y_mean2 = final3rd_Cluster2.y.mean()

                x_end_mean2 = final3rd_Cluster2.endX.mean()
                y_end__mean2 = final3rd_Cluster2.endY.mean()

                final3rd_Cluster.loc[len(final3rd_Cluster.index)] = [club, 'Pass', 'Qualifiers', x_mean0, y_mean0, x_end_mean0, y_end__mean0, 'mean0']
                final3rd_Cluster.loc[len(final3rd_Cluster.index)] = [club, 'Pass', 'Qualifiers', x_mean1, y_mean1, x_end_mean1, y_end__mean1, 'mean1']
                final3rd_Cluster.loc[len(final3rd_Cluster.index)] = [club, 'Pass', 'Qualifiers', x_mean2, y_mean2, x_end_mean2, y_end__mean2, 'mean2']

                #################################################################################################################################################
                
                df_data = df.loc[(df['typedisplayName'] == 'Pass') & (df['outcomeTypedisplayName'] == 'Successful')].reset_index(drop=True)

                xTDF = xT(data)

                DFSides = sides(xTDF, data, club)

                xT_Sides = dataFrame_xTFlow(DFSides)
                
        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(6,4))

        pitch = Pitch(pitch_type='opta',
                    pitch_color='#E8E8E8', line_color='#181818',
                    line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        #################################################################################################################################################

        if matchDay != 'All Season':
                Title = df_data.loc[df_data['Match_ID'] == matchDay]

                home = Title.loc[(Title.team == club)]
                away = Title.loc[(Title.team != club)]
                
                home = home.team.unique()
                homeName = home[0]
                color = color[0]

                away = away.team.unique()
                awayName = away[0]
                color2 = '#ff0000'

        #################################################################################################################################################

        if matchDay != 'All Season':
                #Params for the text inside the <> this is a function to highlight text
                highlight_textprops =\
                        [{"color": color[0],"fontweight": 'bold'},
                        {"color": color2[0],"fontweight": 'bold'}
                        ]

                fig_text(s =f'<{homeName}>' + ' ' + 'vs' + ' ' + f'<{awayName}>',
                         x = 0.53, y = 0.98, ha='center', va='center',
                         highlight_textprops = highlight_textprops ,
                         color='#1b1b1b', fontweight='bold',
                         fontsize=50);
                
                fig_text(s =  'league' + ' ' + '|' + ' ' + 'MatchDay:' + ' ' + str(matchDay) + ' ' + '| Season 21-22 | @menesesp20',
                         x = 0.51, y = 0.94,
                         color='#1b1b1b', fontweight='bold',
                         ha='center', va='center',
                         fontsize=18);

        #################################################################################################################################################

        elif matchDay == 'All Season':
                # Title of our plot
                fig.suptitle(club + ' ' + 'Open Play',
                             fontsize=50, color='#1b1b1b',
                             fontweight = "bold",
                             x=0.525, y=1)

                fig_text(s = "Key Passes | World Cup Catar 2022 | @menesesp20",
                         x = 0.5, y = 0.95,
                         color='#1b1b1b', fontweight='bold',
                         ha='center',
                         fontsize=12);

        #################################################################################################################################################
        # RIGHT
        fig_text(s = str(rightfinal3rdTotal) + ' ' + '%',
                x = 0.77, y = 0.46,
                color='black', fontweight='bold', ha='center' ,fontsize=14);

        # xT Right
        ax.scatter( 14 , 64.3 , marker ='d', lw=2, edgecolor='black', facecolor='None', s = 3000, zorder=3)

        fig_text(s =str(round(xT_Sides.right_xT[0], 2)),
                x = 0.76, y = 0.37,
                color='black', fontweight='bold', ha='center' ,fontsize=8);

        #################################################################################################################################################
        # LEFT
        fig_text(s = str(leftfinal3rdTotal) + ' ' + '%',
                x = 0.292, y = 0.46,
                color='black', fontweight='bold', ha='center' ,fontsize=14);

        # xT Left
        ax.scatter( 83 , 64.3 , marker ='d', lw=2, edgecolor='black', facecolor='None', s = 3000, zorder=3)

        fig_text(s = str(round(xT_Sides.left_xT[0], 2)),
                x = 0.283, y = 0.37,
                color='black', fontweight='bold', ha='center' ,fontsize=8);

        #################################################################################################################################################
        # CENTER
        fig_text(s = str(centerfinal3rdTotal) + ' ' + '%',
                x = 0.525, y = 0.46,
                color='black', fontweight='bold', ha='center' ,fontsize=14);

        # xT Center
        ax.scatter( 49.5 , 64.3 , marker ='d', lw=2, edgecolor='black', facecolor='None', s = 3000, zorder=3)

        fig_text(s = str(round(xT_Sides.center_xT[0], 2)),
                x = 0.515, y = 0.37,
                color='black', fontweight='bold', ha='center' ,fontsize=8);

        #################################################################################################################################################

        left =  str(leftfinal3rdTotal)
        center = str(centerfinal3rdTotal)
        right = str(rightfinal3rdTotal)

        if right > left > center:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1)

        elif right > center > left:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1)

        ##################################################################################################################

        elif left > right > center:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1)


        elif left > center > right:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1)



        ##################################################################################################################

        elif center > left > right:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1)

        elif center > right > left:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1)

        ##################################################################################################################

        elif left == center:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1)

        ##################################################################################################################

        elif left == right:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1)

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 1, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1)
                
        # ADD RECTANGLES
        ax.add_patch(rectangleLeft)
        ax.add_patch(rectangleCenter)
        ax.add_patch(rectangleRight)
        #################################################################################################################################################
        if data == 'WyScout':
                # Key Passes Cluster
                if matchDay == 'All Season':
                        for x in range(len(final3rd_Cluster['cluster'])):
                        
                                if final3rd_Cluster['cluster'][x] == 0:
                                        pitch.lines(xstart=final3rd_Cluster['location.x'][x], ystart=final3rd_Cluster['location.y'][x],
                                                xend=final3rd_Cluster['pass.endLocation.x'][x], yend=final3rd_Cluster['pass.endLocation.y'][x],
                                                color='#ea04dc',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True,
                                                alpha_start=0.2,alpha_end=0.5)

                                        pitch.scatter(final3rd_Cluster['pass.endLocation.x'][x], final3rd_Cluster['pass.endLocation.y'][x],
                                                s = 150,
                                                c='#ea04dc',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3)

                                if final3rd_Cluster['cluster'][x] == 1:
                                        pitch.lines(xstart=final3rd_Cluster['location.x'][x], ystart=final3rd_Cluster['location.y'][x],
                                                xend=final3rd_Cluster['pass.endLocation.x'][x], yend=final3rd_Cluster['pass.endLocation.y'][x],
                                                color='#2d92df',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True,
                                                alpha_start=0.2,alpha_end=0.5)

                                        pitch.scatter(final3rd_Cluster['pass.endLocation.x'][x], final3rd_Cluster['pass.endLocation.y'][x],
                                                s = 150,
                                                c='#2d92df',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3)

                                if final3rd_Cluster['cluster'][x] == 2:
                                        pitch.lines(xstart=final3rd_Cluster['location.x'][x], ystart=final3rd_Cluster['location.y'][x],
                                                xend=final3rd_Cluster['pass.endLocation.x'][x], yend=final3rd_Cluster['pass.endLocation.y'][x],
                                                color='#fb8c04',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True,
                                                alpha_start=0.2,alpha_end=0.5)

                                        pitch.scatter(final3rd_Cluster['pass.endLocation.x'][x], final3rd_Cluster['pass.endLocation.y'][x],
                                                s = 150,
                                                c='#fb8c04',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3)

                                if final3rd_Cluster['cluster'][x] == 'mean0':
                                        pitch.lines(xstart=final3rd_Cluster['location.x'][x], ystart=final3rd_Cluster['location.y'][x],
                                                xend=final3rd_Cluster['pass.endLocation.x'][x], yend=final3rd_Cluster['pass.endLocation.y'][x],
                                                color='#ea04dc',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True)

                                        pitch.scatter(final3rd_Cluster['pass.endLocation.x'][x], final3rd_Cluster['pass.endLocation.y'][x],
                                                s = 150,
                                                c='#ea04dc',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3)

                                if final3rd_Cluster['cluster'][x] == 'mean1':
                                        pitch.lines(xstart=final3rd_Cluster['location.x'][x], ystart=final3rd_Cluster['location.y'][x],
                                                xend=final3rd_Cluster['pass.endLocation.x'][x], yend=final3rd_Cluster['pass.endLocation.y'][x],
                                                color='#2d92df',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True)

                                        pitch.scatter(final3rd_Cluster['pass.endLocation.x'][x], final3rd_Cluster['pass.endLocation.y'][x],
                                                s = 150,
                                                c='#2d92df',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3)

                                if final3rd_Cluster['cluster'][x] == 'mean2':
                                        pitch.lines(xstart=final3rd_Cluster['location.x'][x], ystart=final3rd_Cluster['location.y'][x],
                                                xend=final3rd_Cluster['pass.endLocation.x'][x], yend=final3rd_Cluster['pass.endLocation.y'][x],
                                                color='#fb8c04',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True)

                                        pitch.scatter(final3rd_Cluster['pass.endLocation.x'][x], final3rd_Cluster['pass.endLocation.y'][x],
                                                s = 150,
                                                c='#fb8c04',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3)

        if data == 'WhoScored':
                # Key Passes Cluster
                if matchDay == 'All Season':
                        for x in range(len(final3rd_Cluster['cluster'])):
                        
                                if final3rd_Cluster['cluster'][x] == 0:
                                        pitch.lines(xstart=final3rd_Cluster['x'][x], ystart=final3rd_Cluster['y'][x],
                                                xend=final3rd_Cluster['endX'][x], yend=final3rd_Cluster['endY'][x],
                                                color='#ea04dc',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True,
                                                alpha=0.1)

                                        pitch.scatter(final3rd_Cluster['endX'][x], final3rd_Cluster['endY'][x],
                                                s = 150,
                                                c='#ea04dc',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3,
                                                alpha=0.1)

                                elif final3rd_Cluster['cluster'][x] == 1:
                                        pitch.lines(xstart=final3rd_Cluster['x'][x], ystart=final3rd_Cluster['y'][x],
                                                xend=final3rd_Cluster['endX'][x], yend=final3rd_Cluster['endY'][x],
                                                color='#2d92df',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True,
                                                alpha=0.1)

                                        pitch.scatter(final3rd_Cluster['endX'][x], final3rd_Cluster['endY'][x],
                                                s = 150,
                                                c='#2d92df',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3,
                                                alpha=0.2)

                                elif final3rd_Cluster['cluster'][x] == 2:
                                        pitch.lines(xstart=final3rd_Cluster['x'][x], ystart=final3rd_Cluster['y'][x],
                                                xend=final3rd_Cluster['endX'][x], yend=final3rd_Cluster['endY'][x],
                                                color='#fb8c04',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True,
                                                alpha=0.2)

                                        pitch.scatter(final3rd_Cluster['endX'][x], final3rd_Cluster['endY'][x],
                                                s = 150,
                                                c='#fb8c04',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3,
                                                alpha=0.1)

                                elif final3rd_Cluster['cluster'][x] == 'mean0':
                                        pitch.lines(xstart=final3rd_Cluster['x'][x], ystart=final3rd_Cluster['y'][x],
                                                xend=final3rd_Cluster['endX'][x], yend=final3rd_Cluster['endY'][x],
                                                color='#ea04dc',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True)

                                        pitch.scatter(final3rd_Cluster['endX'][x], final3rd_Cluster['endY'][x],
                                                s = 150,
                                                c='#ea04dc',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3)

                                elif final3rd_Cluster['cluster'][x] == 'mean1':
                                        pitch.lines(xstart=final3rd_Cluster['x'][x], ystart=final3rd_Cluster['y'][x],
                                                xend=final3rd_Cluster['endX'][x], yend=final3rd_Cluster['endY'][x],
                                                color='#2d92df',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True)

                                        pitch.scatter(final3rd_Cluster['endX'][x], final3rd_Cluster['endY'][x],
                                                s = 150,
                                                c='#2d92df',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3)

                                elif final3rd_Cluster['cluster'][x] == 'mean2':
                                        pitch.lines(xstart=final3rd_Cluster['x'][x], ystart=final3rd_Cluster['y'][x],
                                                xend=final3rd_Cluster['endX'][x], yend=final3rd_Cluster['endY'][x],
                                                color='#fb8c04',
                                                ax=ax,
                                                zorder=2,
                                                comet=True,
                                                transparent=True)

                                        pitch.scatter(final3rd_Cluster['endX'][x], final3rd_Cluster['endY'][x],
                                                s = 150,
                                                c='#fb8c04',
                                                edgecolor='#ffffff',
                                                ax=ax,
                                                zorder=3)
        #################################################################################################################################################

        fig_text(s = 'Most frequent zone',
                 x = 0.34, y = 0.88,
                 color='#ea04dc', fontweight='bold', ha='center' ,fontsize=5);

        fig_text(s = 'Second most frequent zone',
                 x = 0.45, y = 0.88,
                 color='#2d92df', fontweight='bold', ha='center' ,fontsize=5);

        fig_text(s = 'Third most frequent zone',
                 x = 0.57, y = 0.88,
                 color='#fb8c04', fontweight='bold', ha='center' ,fontsize=5);

        #fig_text(s = 'Coach: Jorge Jesus',
        #         x = 0.223, y = 0.86,
        #         color='#181818', fontweight='bold', ha='center', alpha=0.8, fontsize=12);

        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.07, bottom=0.85, width=0.05, height=0.1)

        # END NOTE
        fig_text(s = 'The values inside the diamond are the xT value for each third',
                 x = 0.5, y = 0.125,
                 color='#1b1b1b', fontweight='bold', ha='center' ,fontsize=5);

        fig_text(s = 'xT values based on Karun Singhs model',
                 x = 0.765, y = 0.875,
                 color='#1b1b1b', fontweight='bold', ha='center' ,fontsize=5);

        plt.savefig('assets/finalThird' + club + '.png', dpi=300)

        return app.get_asset_url('finalThird' + club + '.png')

################################################################################################################################################

def cornersTaken(club, data):

        if 'level_0' in df.columns:
                df.drop(['level_0'], axis=1, inplace=True)
        else:
                pass
        
        cornersData = []
        
        if data == 'WhoScored':
                
                df_Corner = search_qualifierOPTA(cornersData, 'CornerTaken')

                right_corner = df_Corner.loc[df_Corner['y'] < 50]

                left_corner = df_Corner.loc[df_Corner['y'] > 50]

        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(6,4))

        pitch = VerticalPitch(pitch_type='opta', half=True,
                    pitch_color='#E8E8E8', line_color='#181818',
                    line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        #################################################################################################################################################

        # Title of our plot - WITH ANGLES BOTTOM: 0.98, 0.93

        fig.suptitle(club + ' ' + 'Corners', fontsize=14, color='#181818', fontweight = "bold", x=0.5, y=0.93, ha='center', va='center')

        Title = fig_text(s = 'World Cup Catar 2022 | @menesesp20',
                         x = 0.5, y = 0.89,
                         color='#181818', fontweight='bold', ha='center', va='center', fontsize=5);

        #################################################################################################################################################

        if data == 'WhoScored':
                firstCorner_L_Cluster = cluster_Event(left_corner, club, 'CornerTaken', 3, data)

                firstCorner_L_Cluster['cluster'].value_counts().reset_index(drop=True)

                #################################################################################################################################################

                firstCorner_R_Cluster = cluster_Event(right_corner, club, 'CornerTaken', 3, data)

                firstCorner_R_Cluster['cluster'].value_counts().reset_index(drop=True)
                
                print(firstCorner_R_Cluster['cluster'].value_counts().reset_index(drop=True))
                
        #################################################################################################################################################

        if data == 'WhoScored':
                # RIGHT SIDE CLUSTER
                for x in range(len(firstCorner_R_Cluster['cluster'])):

                        if firstCorner_R_Cluster['cluster'][x] == 0:
                                #Criação das setas que simbolizam os passes realizados falhados
                                pitch.lines(firstCorner_R_Cluster['x'][x], firstCorner_R_Cluster['y'][x],
                                        firstCorner_R_Cluster['endX'][x], firstCorner_R_Cluster['endY'][x],
                                        color='#ea04dc',
                                        ax=ax,
                                        zorder=3,
                                        comet=True,
                                        transparent=True,
                                        alpha_start=0.2,alpha_end=0.8)
                        
                                pitch.scatter(firstCorner_R_Cluster['endX'][x], firstCorner_R_Cluster['endY'][x],
                                        s = 30,
                                        marker='o',
                                        c='#1b1b1b',
                                        edgecolor='#ea04dc',
                                        ax=ax,
                                        zorder=4)


        # CIRCLE                            
        ax.scatter( 40 , 95 , s = 1500, color='#eb00e5', alpha=0.5, lw=1)

        ax.annotate('', xy=(18, 84), xytext=(5, 84),
                size=14, color = '#eb00e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#eb00e5', lw=1))

        fig_text(s = 'Most frequent zone',
                 x = 0.75, y = 0.66,
                 color='#eb00e5', fontweight='bold', ha='center', va='center', fontsize=5);

        #################################################################################################################################################

        if data == 'WhoScored':
                # LEFT SIDE CLUSTER
                for x in range(len(firstCorner_L_Cluster['cluster'])):        
                        if firstCorner_L_Cluster['cluster'][x] == 1:
                                #Criação das setas que simbolizam os passes realizados falhados
                                pitch.lines(firstCorner_L_Cluster['x'][x], firstCorner_L_Cluster['y'][x],
                                        firstCorner_L_Cluster['endX'][x], firstCorner_L_Cluster['endY'][x],
                                        color='#2d92df',
                                        ax=ax,
                                        zorder=3,
                                        comet=True,
                                        transparent=True,
                                        alpha_start=0.2,alpha_end=0.8)
                        
                                pitch.scatter(firstCorner_L_Cluster['endX'][x], firstCorner_L_Cluster['endY'][x],
                                        s = 30,
                                        marker='o',
                                        c='#1b1b1b',
                                        edgecolor='#2d92df',
                                        ax=ax,
                                        zorder=4)
                
        # CIRCLE                            
        ax.scatter(60, 95, s = 1500, color='#2894e5', alpha=0.5, lw=1)

        ax.annotate('', xy=(83, 84), xytext=(95, 84),
                size=14, color = '#2894e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#2894e5', lw=1))

        fig_text(s = 'Most frequent zone',
                 x = 0.273, y = 0.66,
                 color='#2894e5', fontweight='bold', ha='center', va='center', fontsize=5);

        #################################################################################################################################################

        # PENTAGON RIGHT                          
        ax.scatter(40, 65, marker = 'p', s = 1500, color='#eb00e5', alpha=0.5, lw=1)

        fig_text(s =  str(len(firstCorner_R_Cluster)),
                        x = 0.572, y = 0.378,
                        color='#181818', fontweight='bold', ha='center', fontsize=10);

        #################################################################################################################################################

        # PENTAGON LEFT                           
        ax.scatter( 60 , 65 , marker = 'p', s = 1500, color='#2894e5', alpha=0.5, lw=1)

        fig_text(s = str(len(firstCorner_L_Cluster)),
                 x = 0.45, y = 0.378,
                 color='#181818', fontweight='bold', ha='center', fontsize=10);

        #################################################################################################################################################

        # Club Logo - WITH ANGLES BOTTOM: 0.89, LEFT:0.14
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.08, bottom=0.85, width=0.05, height=0.1)

        #################################################################################################################################################

        # Angle Left Logo
        #fig = add_image(image='angleLeft.png', fig=fig, left=0.082, bottom=0.842, width=0.2, height=0.1)

        # ANGLE LEFT VALUE
        #fig_text(s = '4.6°',
        #                x = 0.179, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        #################################################################################################################################################

        # Angle Right Logo
        #fig = add_image(image='angleRight.png', fig=fig, left=0.7425, bottom=0.842, width=0.2, height=0.1)

        # ANGLE RIGHT VALUE
        #fig_text(s = '1.8°',
        #                x = 0.846, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        fig_text(s = 'The values inside pentagon are the total of corners made by each side',
                x = 0.42, y = 0.129,
                color='#181818', fontweight='bold', ha='center' ,fontsize=5);

        plt.savefig('assets/corners' + club + '.png', dpi=300)

        return app.get_asset_url('corners' + club + '.png')

################################################################################################################################################

def corners1stPostTaken(club):
        
        if 'level_0' in df.columns:
                df.drop(['level_0'], axis=1, inplace=True)
        else:
                pass
        
        cornersData = []

        df_Corner = df.loc[df['type.primary'] == 'corner'].reset_index(drop=True)

        right_corner = df_Corner.loc[df_Corner['location.y'] < 50]

        left_corner = df_Corner.loc[df_Corner['location.y'] > 50]

        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(18,14))

        pitch = VerticalPitch(pitch_type='opta',
                              pitch_color='#1b1b1b', line_color='white', half = True,
                              line_zorder=1, linewidth=5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#1b1b1b')

        #################################################################################################################################################

        # Title of our plot - WITH ANGLES BOTTOM: 0.98, 0.93

        fig.suptitle(club + ' ' + '1st Post Corners', fontsize=40, color='white',
                      fontweight = "bold", x=0.525, y=0.955)

        Title = fig_text(s = 'Season 21-22 | Made by: @Menesesp20',
                         x = 0.5, y = 0.91,
                         color='white', fontweight='bold', ha='center' ,fontsize=16);

        #################################################################################################################################################

        firstCorner_L = left_corner.loc[(left_corner['pass.endLocation.y'] >= 55) & (left_corner['pass.endLocation.y'] <= 79)]

        firstCorner_L_Cluster = cluster_Event(firstCorner_L, club, 'corner', 3)

        firstCorner_L_Cluster['cluster'].value_counts().reset_index(drop=True)

        #################################################################################################################################################

        firstCorner_R = right_corner.loc[(right_corner['pass.endLocation.y'] <= 45) & (right_corner['pass.endLocation.y'] >= 21)]

        firstCorner_R_Cluster = cluster_Event(firstCorner_R, club, 'corner', 3)

        firstCorner_R_Cluster['cluster'].value_counts().reset_index(drop=True)

        #################################################################################################################################################

        # RIGHT SIDE CLUSTER
        for x in range(len(firstCorner_R_Cluster['cluster'])):

                if firstCorner_R_Cluster['cluster'][x] == 1:
                        #Criação das setas que simbolizam os passes realizados falhados
                        pitch.lines(firstCorner_R_Cluster['location.x'][x], firstCorner_R_Cluster['location.y'][x],
                                    firstCorner_R_Cluster['pass.endLocation.x'][x], firstCorner_R_Cluster['pass.endLocation.y'][x],
                                    color='#ea04dc',
                                    ax=ax,
                                    zorder=3,
                                    comet=True,
                                    transparent=True,
                                    alpha_start=0.2,alpha_end=0.8)
                
                        pitch.scatter(firstCorner_R_Cluster['pass.endLocation.x'][x], firstCorner_R_Cluster['pass.endLocation.y'][x],
                                      s = 100,
                                      marker='o',
                                      c='#1b1b1b',
                                      edgecolor='#ea04dc',
                                      ax=ax,
                                      zorder=4)
        # CIRCLE                            
        ax.scatter( 40 , 95 , s = 5000, color='#eb00e5', alpha=0.5, lw=2)

        ax.annotate('', xy=(18, 84), xytext=(5, 84),
                size=14, color = '#eb00e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#eb00e5', lw=2))

        fig_text(s = 'Most frequent zone',
                x = 0.794, y = 0.66,
                color='#eb00e5', fontweight='bold', ha='center' ,fontsize=5);

        #################################################################################################################################################

        # LEFT SIDE CLUSTER
        for x in range(len(firstCorner_L_Cluster['cluster'])):        
                if firstCorner_L_Cluster['cluster'][x] == 0:
                        #Criação das setas que simbolizam os passes realizados falhados
                        pitch.lines(firstCorner_L_Cluster['location.x'][x], firstCorner_L_Cluster['location.y'][x],
                                    firstCorner_L_Cluster['endX'][x], firstCorner_L_Cluster['endY'][x],
                                    color='#2d92df',
                                    ax=ax,
                                    zorder=3,
                                    comet=True,
                                    transparent=True,
                                    alpha_start=0.2,alpha_end=0.8)
                
                        pitch.scatter(firstCorner_L_Cluster['pass.endLocation.x'][x], firstCorner_L_Cluster['pass.endLocation.y'][x],
                                      s = 30,
                                      marker='o',
                                      c='#1b1b1b',
                                      edgecolor='#2d92df',
                                      ax=ax,
                                      zorder=4)
        # CIRCLE                            
        ax.scatter( 60 , 95 , s = 20000, color='#2894e5', alpha=0.5, lw=2)

        ax.annotate('', xy=(83, 84), xytext=(95, 84),
                size=14, color = '#2894e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#2894e5', lw=2))

        fig_text(s = 'Most frequent zone',
                x = 0.23, y = 0.66,
                color='#2894e5', fontweight='bold', ha='center' ,fontsize=5);
        

        #################################################################################################################################################

        # PENTAGON RIGHT                          
        ax.scatter( 40 , 65 , marker = 'p', s = 5000, color='#eb00e5', alpha=0.5, lw=2)

        # VALUE FIRST CORNER MOST FREQUENT ON RIGHT SIDE

        firstCornerR =  int((len(firstCorner_R) / len(right_corner) * 100))

        fig_text(s =  str(firstCornerR) + '%',
                        x = 0.584, y = 0.378,
                        color='white', fontweight='bold', ha='center' ,fontsize=8);

        #################################################################################################################################################

        # PENTAGON LEFT                           
        ax.scatter( 60 , 65 , marker = 'p', s = 5000, color='#2894e5', alpha=0.5, lw=2)

        # VALUE FIRST CORNER MOST FREQUENT ON LEFT SIDE

        firstCornerL = int((len(firstCorner_L) / len(left_corner) * 100))

        fig_text(s = str(firstCornerL) + '%',
                        x = 0.44, y = 0.378,
                        color='white', fontweight='bold', ha='center' ,fontsize=8);

        #################################################################################################################################################

        # Club Logo - WITH ANGLES BOTTOM: 0.89, LEFT:0.14
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.12, bottom=0.85, width=0.05, height=0.1)

        #################################################################################################################################################

        # Angle Left Logo
        #fig = add_image(image='C:/Users/menes/Documents/Data Hub/angleLeft.png', fig=fig, left=0.082, bottom=0.842, width=0.2, height=0.1)

        # ANGLE LEFT VALUE
        #fig_text(s = '4.6°',
        #                x = 0.179, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        #################################################################################################################################################

        # Angle Right Logo
        #fig = add_image(image='C:/Users/menes/Documents/Data Hub/angleRight.png', fig=fig, left=0.7425, bottom=0.842, width=0.2, height=0.1)

        # ANGLE RIGHT VALUE
        #fig_text(s = '1.8°',
        #                x = 0.846, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        fig_text(s = 'The values inside pentagon are the percentage of corners made by each side for the circle area',
                x = 0.407, y = 0.14,
                color='white', fontweight='bold', ha='center' ,fontsize=5);

        plt.savefig('assets/corners1st' + club + '.png', dpi=300)

        return app.get_asset_url('corners1st' + club + '.png')

################################################################################################################################################

def corners2ndPostTaken(club):
        
        if 'level_0' in df.columns:
                df.drop(['level_0'], axis=1, inplace=True)
        else:
                pass
        
        cornersData = []

        df_Corner = df.loc[df['type.primary'] == 'corner'].reset_index(drop=True)

        right_corner = df_Corner.loc[df_Corner['location.y'] < 50]

        left_corner = df_Corner.loc[df_Corner['location.y'] > 50]

        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(18,14))

        pitch = VerticalPitch(pitch_type='opta',
                              pitch_color='#1b1b1b', line_color='white', half = True,
                              line_zorder=1, linewidth=5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#1b1b1b')

        #################################################################################################################################################

        # Title of our plot - WITH ANGLES BOTTOM: 0.98, 0.93

        fig.suptitle(club + ' ' + '2nd Post Corners', fontsize=40, color='white',
        fontweight = "bold", x=0.525, y=0.955)

        Title = fig_text(s = 'Season 21-22 | Made by: @Menesesp20',
                        x = 0.5, y = 0.91,
                        color='white', fontweight='bold', ha='center' ,fontsize=16);

        #################################################################################################################################################

        secondCorner_L = left_corner.loc[(left_corner['pass.endLocation.y'] <= 55) & (left_corner['pass.endLocation.y'] >= 21) & (left_corner['pass.endLocation.x'] >= 90)]
        if secondCorner_L.shape[0] == 0:
                pass
        else:
                secondCorner_L_Cluster = cluster_Event(secondCorner_L, club, 'corner', 2)

                secondCorner_L_Cluster['cluster'].value_counts().reset_index(drop=True)

                # LEFT SIDE CLUSTER
                for x in range(len(secondCorner_L_Cluster['cluster'])):        
                        if secondCorner_L_Cluster['cluster'][x] == 0:
                                #Criação das setas que simbolizam os passes realizados falhados
                                pitch.lines(secondCorner_L_Cluster['location.x'][x], secondCorner_L_Cluster['location.y'][x],
                                        secondCorner_L_Cluster['pass.endLocation.x'][x], secondCorner_L_Cluster['pass.endLocation.y'][x],
                                        color='#ea04dc',
                                        ax=ax,
                                        zorder=3,
                                        comet=True,
                                        transparent=True,
                                        alpha_start=0.2,alpha_end=0.8)
                        
                                pitch.scatter(secondCorner_L_Cluster['pass.endLocation.x'][x], secondCorner_L_Cluster['pass.endLocation.y'][x],
                                        s = 100,
                                        marker='o',
                                        c='#1b1b1b',
                                        edgecolor='#ea04dc',
                                        ax=ax,
                                        zorder=4)
                
                # CIRCLE 2nd Post                           
                ax.scatter( 40 , 95 , s = 20000, color='#2894e5', alpha=0.5, lw=3)

                # PENTAGON LEFT                           
                ax.scatter( 60 , 65 , marker = 'p', s = 20000, color='#2894e5', alpha=0.5, lw=3)

                len2ndCornerL = len(secondCorner_L_Cluster.loc[secondCorner_L_Cluster['cluster']==0])

                secondCornerL = int((len(secondCorner_L) / len(left_corner) * 100))

                fig_text(s = str(secondCornerL) + '%',
                                x = 0.44, y = 0.378,
                                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=28);

        #################################################################################################################################################

        secondCorner_R = right_corner.loc[(right_corner['pass.endLocation.y'] <= 75) & (right_corner['pass.endLocation.y'] >= 55) & (right_corner['pass.endLocation.x'] >= 90)]
        if secondCorner_R.shape[0] == 0:
                pass
        else:
                secondCorner_R_Cluster = cluster_Event(secondCorner_R, club, 'corner', 3)
                
                secondCorner_R_Cluster['cluster'].value_counts().reset_index(drop=True)

                # RIGHT SIDE CLUSTER
                for x in range(len(secondCorner_R_Cluster['cluster'])):

                        if secondCorner_R_Cluster['cluster'][x] == 1:
                                #Criação das setas que simbolizam os passes realizados falhados
                                pitch.lines(secondCorner_R_Cluster['location.x'][x], secondCorner_R_Cluster['location.y'][x],
                                        secondCorner_R_Cluster['pass.endLocation.x'][x], secondCorner_R_Cluster['pass.endLocation.y'][x],
                                        color='#2d92df',
                                        ax=ax,
                                        zorder=3,
                                        comet=True,
                                        transparent=True,
                                        alpha_start=0.2,alpha_end=0.8)
                        
                                pitch.scatter(secondCorner_R_Cluster['pass.endLocation.x'][x], secondCorner_R_Cluster['pass.endLocation.y'][x],
                                        s = 100,
                                        marker='o',
                                        c='#1b1b1b',
                                        edgecolor='#2d92df',
                                        ax=ax,
                                        zorder=4)
                # CIRCLE 1st Post                           
                ax.scatter( 60 , 95 , s = 20000, color='#eb00e5', alpha=0.5, lw=3)            

                # PENTAGON RIGHT                          
                ax.scatter( 40 , 65 , marker = 'p', s = 20000, color='#eb00e5', alpha=0.5, lw=3)

                len2ndCornerR = len(secondCorner_R_Cluster.loc[secondCorner_R_Cluster['cluster']==0])

                secondCornerR = int((len(secondCorner_R) / len(right_corner) * 100))

                fig_text(s =  str(secondCornerR) + '%',
                                x = 0.584, y = 0.378,
                                color='white', fontweight='bold', ha='center' ,fontsize=30);


        #################################################################################################################################################

        # MOST FREQUENT ZONES ARROWS
        ax.annotate('', xy=(18, 84), xytext=(5, 84),
                size=14, color = '#eb00e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#eb00e5', lw=3))

        fig_text(s = 'Most frequent zone',
                x = 0.794, y = 0.66,
                color='#eb00e5', fontweight='bold', ha='center' ,fontsize=12);

        #################################################################################################################################################

        # MOST FREQUENT ZONES ARROWS
        ax.annotate('', xy=(83, 84), xytext=(95, 84),
                size=14, color = '#2894e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#2894e5', lw=3))

        fig_text(s = 'Most frequent zone',
                x = 0.23, y = 0.66,
                color='#2894e5', fontweight='bold', ha='center' ,fontsize=12);

        #################################################################################################################################################

        # Club Logo - WITH ANGLES BOTTOM: 0.89, LEFT:0.14
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.12, bottom=0.87, width=0.2, height=0.1)

        #################################################################################################################################################

        # Angle Left Logo
        #fig = add_image(image='C:/Users/menes/Documents/Data Hub/angleLeft.png', fig=fig, left=0.082, bottom=0.842, width=0.2, height=0.1)

        # ANGLE LEFT VALUE
        #fig_text(s = '4.6°',
        #                x = 0.179, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        #################################################################################################################################################

        # Angle Right Logo
        #fig = add_image(image='C:/Users/menes/Documents/Data Hub/angleRight.png', fig=fig, left=0.7425, bottom=0.842, width=0.2, height=0.1)

        # ANGLE RIGHT VALUE
        #fig_text(s = '1.8°',
        #                x = 0.846, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        fig_text(s = 'The values inside pentagon are the percentage of corners made by each side for the circle area',
                x = 0.407, y = 0.129,
                color='white', fontweight='bold', ha='center' ,fontsize=12);

        plt.savefig('assets/corners2nd' + club + '.png', dpi=300)

        return app.get_asset_url('corners2nd' + club + '.png')

################################################################################################################################################

def SetPiece_throwIn(club, match=None):

        if 'level_0' in df.columns:
                df.drop(['level_0'], axis=1, inplace=True)
        else:
                pass

        throwIn = []

        throwIn = df.loc[df['type.primary'] == 'throw_in'].reset_index(drop=True)

        if match != None:
                match = df.loc[df.Match_ID == match]
        else:
                match = df.copy()

        #################################################################################################################################################

        # DEFEND SIDE
        defendLeft = match.loc[(match['location.x'] < 35) & (match['location.y'] > 50)]

        defendRight = match.loc[(match['location.x'] < 35) & (match['location.y'] < 50)]

        # MIDDLE SIDE
        middleLeft = match.loc[(match['location.x'] > 35) & (match['location.x'] < 65) & (match['location.y'] > 50)]

        middleRight = match.loc[(match['location.x'] > 35) & (match['location.x'] < 65) & (match['location.y'] < 50)]

        # ATTACK SIDE
        attackLeft = match.loc[(match['location.x'] > 65) & (match['location.y'] > 50)]

        attackRight = match.loc[(match['location.x'] > 65) & (match['location.y'] < 50)]

        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(21,15))

        pitch = VerticalPitch(pitch_type='opta',
                              pitch_color='#E8E8E8', line_color='#181818',
                              line_zorder=1, linewidth=5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        #################################################################################################################################################

        # Title of our plot - WITH ANGLES BOTTOM: 0.98, 0.93

        fig.suptitle(club + ' ' + "Throw-In's", fontsize=45, color='#181818',
                     fontweight = "bold", x=0.545, y=0.955)

        Title = fig_text(s = 'Season 21-22 | Made by: @Menesesp20',
                         x = 0.54, y = 0.91,
                         color='#181818', fontweight='bold', ha='center' ,fontsize=14);

        #################################################################################################################################################
        # DEFEND SIDE CLUSTER
        defendLeft_Cluster = cluster_Event(defendLeft, club, 'throw_in', 2)

        defendLeft_Cluster['cluster'].value_counts().reset_index(drop=True)

        defendRight_Cluster = cluster_Event(defendRight, club, 'throw_in', 3)

        defendRight_Cluster['cluster'].value_counts().reset_index(drop=True)

        #################################################################################################################################################

        # MIDDLE SIDE CLUSTER
        middleLeft_Cluster = cluster_Event(middleLeft, club, 'throw_in', 1)

        middleLeft_Cluster['cluster'].value_counts().reset_index(drop=True)

        middleRight_Cluster = cluster_Event(middleRight, club, 'throw_in', 3)

        middleRight_Cluster['cluster'].value_counts().reset_index(drop=True)

        #################################################################################################################################################

        # ATTACK SIDE CLUSTER
        attackLeft_Cluster = cluster_Event(attackLeft, club, 'throw_in', 2)

        attackLeft_Cluster['cluster'].value_counts().reset_index(drop=True)

        attackRight_Cluster = cluster_Event(attackRight, club, 'throw_in', 3)

        attackRight_Cluster['cluster'].value_counts().reset_index(drop=True)

        ####################################################################################################################################################
        # DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND
        #################################################################################################################################################
        if defendLeft_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(defendLeft_Cluster['cluster'])):
                        
                        if defendLeft_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=defendLeft_Cluster['location.x'][x], ystart=defendLeft_Cluster['location.y'][x],
                                        xend=defendLeft_Cluster['pass.endLocation.x'][x], yend=defendLeft_Cluster['pass.endLocation.y'][x],
                                        color='#eb00e5',
                                        lw=3, zorder=2,
                                        ax=ax)
        ####################################################################################################################################################
        # DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND
        ####################################################################################################################################################

        if defendRight_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(defendRight_Cluster['cluster'])):
                        
                        if defendRight_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=defendRight_Cluster['location.x'][x], ystart=defendRight_Cluster['location.y'][x],
                                        xend=defendRight_Cluster['pass.endLocation.x'][x], yend=defendRight_Cluster['pass.endLocation.y'][x],
                                        color='#2894e5',
                                        lw=3, zorder=2,
                                        ax=ax)

        ####################################################################################################################################################
        # MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE
        ####################################################################################################################################################

        if middleLeft_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(middleLeft_Cluster['cluster'])):
                        
                        if middleLeft_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=middleLeft_Cluster['location.x'][x], ystart=middleLeft_Cluster['location.y'][x],
                                        xend=middleLeft_Cluster['pass.endLocation.x'][x], yend=middleLeft_Cluster['pass.endLocation.y'][x],
                                        color='#ffe506',
                                        lw=3, zorder=2,
                                        ax=ax)

        ####################################################################################################################################################
        # MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE
        ####################################################################################################################################################

        if middleRight_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(middleRight_Cluster['cluster'])):
                        
                        if middleRight_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=middleRight_Cluster['location.x'][x], ystart=middleRight_Cluster['location.y'][x],
                                        xend=middleRight_Cluster['pass.endLocation.x'][x], yend=middleRight_Cluster['pass.endLocation.y'][x],
                                        color='#ffe506',
                                        lw=3, zorder=2,
                                        ax=ax)

        ####################################################################################################################################################
        # ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK
        ####################################################################################################################################################
        if attackLeft_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(attackLeft_Cluster['cluster'])):
                        
                        if attackLeft_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=attackLeft_Cluster['location.x'][x], ystart=attackLeft_Cluster['location.y'][x],
                                        xend=attackLeft_Cluster['pass.endLocation.x'][x], yend=attackLeft_Cluster['pass.endLocation.y'][x],
                                        color='#eb00e5',
                                        lw=3, zorder=2,
                                        ax=ax)
                                        
        #################################################################################################################################################
        # ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK
        #################################################################################################################################################

        if attackRight_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(attackRight_Cluster['cluster'])):
                        
                        if attackRight_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=attackRight_Cluster['location.x'][x], ystart=attackRight_Cluster['location.y'][x],
                                        xend=attackRight_Cluster['pass.endLocation.x'][x], yend=attackRight_Cluster['pass.endLocation.y'][x],
                                        color='#2894e5',
                                        lw=3, zorder=2,
                                        ax=ax)

        #################################################################################################################################################

        fig_text(s = 'Blue - Right Side',
                x = 0.648, y = 0.12,
                color='#2894e5', fontweight='bold', ha='center' ,fontsize=12);

        fig_text(s = 'Purple - Left Side',
                x = 0.38, y = 0.12,
                color='#eb00e5', fontweight='bold', ha='center' ,fontsize=12);

        fig_text(s = 'Yellow - Middle Side',
                x = 0.518, y = 0.12,
                color='#ffe506', fontweight='bold', ha='center' ,fontsize=12);

        #################################################################################################################################################

        ax.axhline(35,c='#181818', ls='--', lw=4)
        ax.axhline(65,c='#181818', ls='--', lw=4)

        #################################################################################################################################################

        # ATTACK
        #fig_text(s = '12',
        #        x = 0.512, y = 0.683,
        #        fontfamily = 'medium', color='Black', fontweight='bold', ha='center' ,fontsize=30);

        #ax.scatter( 50 , 27 , marker = 'p', s = 12000, color='#181818', alpha=0.8, lw=3)

        # MIDDLE

        #fig_text(s = '12',
        #        x = 0.512, y = 0.518,
        #        fontfamily = 'medium', color='Black', fontweight='bold', ha='center' ,fontsize=30);

        #ax.scatter( 50 , 50 , marker = 'p', s = 12000, color='#181818', alpha=0.8, lw=3)

        # DEFENSE

        #fig_text(s = '12',
        #        x = 0.512, y = 0.348,
        #        fontfamily = 'medium', color='Black', fontweight='bold', ha='center' ,fontsize=30);

        #ax.scatter( 50 , 72 , marker = 'p', s = 12000, color='#181818', alpha=0.8, lw=3)

        # Club Logo - WITH ANGLES BOTTOM: 0.89, LEFT:0.14
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig, left=0.23, bottom=0.85, width=0.05, height=0.07)

        plt.savefig('assets/throwIn' + club + '.png', dpi=300)

        return app.get_asset_url('throwIn' + club + '.png')

################################################################################################################################################

def field_Tilt(club, gameDay):

    if gameDay == 'All Season':
        touch = df.loc[(df.team == club) & (df['typedisplayName'] == 'Pass') & (df['outcomeTypedisplayName'] == 'Successful') & (df['x'] >=75)].reset_index(drop=True)
        
    elif gameDay != 'All Season':
        touch = df.loc[(df['Match_ID'] == gameDay) & (df['typedisplayName'] == 'Pass') & (df['isTouch'] == True) & (df['x'] >= 75)].reset_index(drop=True)

    #############################################################################################################################################

    home = touch['home_Team'].unique()
    home = home[0]
    color = clubColors.get(home)

    away = touch['away_Team'].unique()
    away = away[0]
    color2 = clubColors.get(away)

    home_Passes = touch.loc[(touch['isTouch'] == True) & (touch['team'] == home)]['typedisplayName'].count()
    away_Passes = touch.loc[(touch['isTouch'] == True) & (touch['team'] == away)]['typedisplayName'].count()

    passes_Total = touch.loc[(touch['isTouch'] == True)]['typedisplayName'].count()

    home_Passes = int(home_Passes)
    home_Passes = round((home_Passes / int(passes_Total)) * 100, 2)
    
    away_Passes = int(away_Passes)
    away_Passes = round((away_Passes / int(passes_Total)) * 100, 2)

    #############################################################################################################################################


    fieldTilt_Home = touch.loc[touch['team'] == home]

    fieldTilt_Home = round((len(fieldTilt_Home) / len(touch)) * 100, 2)

    fieldTilt_Away = touch.loc[touch['team'] == away]

    fieldTilt_Away = round((len(fieldTilt_Away) / len(touch)) * 100, 2)

    #############################################################################################################################################

    # Plotting the pitch

    fig, ax = plt.subplots(figsize=(6,4))

    pitch = Pitch(pitch_type='opta',
                    pitch_color='#E8E8E8', line_color='#181818',
                    line_zorder=3, linewidth=0.5, spot_scale=0.00)

    pitch.draw(ax=ax)

    fig.set_facecolor('#E8E8E8')

    #############################################################################################################################################

    ax.axvspan(75, 100, facecolor=color[0], alpha=0.68)

    ax.axvline(75, c='#181818', ls='--', lw=2)


    ax.axvspan(25, 0, facecolor='#ff0000', alpha=0.68)

    ax.axvline(25, c='#181818', ls='--', lw=2)

    #############################################################################################################################################

    for i in range(len(touch)):
        if touch['team'].values[i] == home:
            ax.scatter(touch['x'] , touch['y'] , s = 30, color=color[0], edgecolor='#181818', alpha=0.8, zorder=5)
            
        elif touch['team'].values[i] == away:
            ax.scatter(100 - touch['x'].values[i] , 100 - touch['y'].values[i] , s = 30, color=color2[0], edgecolor='#181818', alpha=0.8, zorder=5)

    #############################################################################################################################################

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
        [{"color": color[0],"fontweight": 'bold'},
         {"color": color2[0],"fontweight": 'bold'}
         ]

    fig_text(s =f'<{home}>' + ' ' + 'vs' + ' ' + f'<{away}>',
             x = 0.515, y = 0.96,
             ha='center', va='center',
             highlight_textprops = highlight_textprops, 
             color='#181818', fontweight='bold',
             fontsize=14);
    
    fig_text(s =  'World Cup Catar 2022 | @menesesp20',
             x = 0.515, y = 0.93,
             color='#181818', fontweight='bold',
             ha='center', va='center',
             fontsize=4);

    fig_text(s = str(fieldTilt_Home) + ' ',
             x = 0.474, y = 0.225,
             color=color[0], fontweight='bold',
             ha='center', va='center',
             fontsize=7)

    fig_text(s = ' ' + '   ' + ' ',
             x = 0.512, y = 0.225,
             color=color2[0], fontweight='bold',
             ha='center', va='center',
             fontsize=7)
    
    fig_text(s = ' ' + str(fieldTilt_Away),
             x = 0.55, y = 0.225,
             color=color2[0], fontweight='bold',
             ha='center', va='center',
             fontsize=7)


    if (home_Passes < 50) & (fieldTilt_Home > 50):
        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
            [{"color": color[0],"fontweight": 'bold'},
            {"color": color[0],"fontweight": 'bold'},
            {"color": color2[0],"fontweight": 'bold'}  + '\n' ]

        fig_text(s = 'Despite' + ' ' + f'<{home}>' + ' ' + 'had less possession' + ' ' + '(' + f'<{str(home_Passes)}%>' + ')' + '\n' +
                 'they had greater ease in penetrating' + '\n' + 'the final third than' + ' ' +  f'<{away}>',
                 highlight_textprops = highlight_textprops,
                 x = 0.528, y = 0.88,
                 color='#181818', fontweight='bold',
                 ha='center', va='center',
                 fontsize=6)

    elif (away_Passes < 50) & (fieldTilt_Away > 50):
        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
            [{"color": color2[0],"fontweight": 'bold'},
            {"color": color2[0],"fontweight": 'bold'},
            {"color": color[0],"fontweight": 'bold'}]

        fig_text(s = 'Despite' + ' ' + f'<{away}>' + ' ' + 'had less possession' + ' ' + '(' + f'<{str(away_Passes)}%>' + ')' + '\n' +
                 'they had greater ease in penetrating' + '\n' + 'the final third than' + ' ' +  f'<{home}>',
                 highlight_textprops = highlight_textprops,
                 x = 0.528, y = 0.88,
                 color='#181818', fontweight='bold',
                 ha='center', va='center',
                 fontsize=6)

    elif (home_Passes > 50) & (fieldTilt_Home < 50):
        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
            [{"color": color[0],"fontweight": 'bold'},
            {"color": color[0],"fontweight": 'bold'},
            {"color": color2[0],"fontweight": 'bold'}]

        fig_text(s = 'Despite' + ' ' + f'<{home}>' + ' ' + 'had more possession' + ' ' + '(' + f'<{str(home_Passes)}%>' + ')' + '\n' +
                 'they struggled to penetrate' + '\n' + 'the last third than' + ' ' +  f'<{away}>',
                 highlight_textprops = highlight_textprops,
                 x = 0.528, y = 0.88,
                 color='#181818', fontweight='bold',
                 ha='center', va='center',
                 fontsize=6)

    elif (away_Passes > 50) & (fieldTilt_Away < 50):
        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
            [{"color": color2[0],"fontweight": 'bold'},
            {"color": color2[0],"fontweight": 'bold'},
            {"color": color[0],"fontweight": 'bold'}]

        fig_text(s = 'Despite' + ' ' + f'<{away}>' + ' ' + 'had more possession' + ' ' + '(' + f'<{str(away_Passes)}%>' + ')' + '\n' +
                 'they struggled to penetrate' + '\n' + 'the last third than' + ' ' +  f'<{home}>',
                 highlight_textprops = highlight_textprops,
                 x = 0.528, y = 0.88,
                 color='#181818', fontweight='bold',
                 ha='center', va='center',
                 fontsize=6)

    elif (fieldTilt_Home > fieldTilt_Away):
        fig_text(s = f'<{home}>' + ' ' + 'dominated the game with greater dominance' + '\n' + 'of the last third than their opponent' + ' ' + 
                    f'<{away}>.',
                    highlight_textprops = highlight_textprops,
                    x = 0.528, y = 0.88,
                    color='#181818', fontweight='bold',
                    ha='center', va='center',
                    fontsize=5)

    elif (fieldTilt_Home < fieldTilt_Away):
        highlight_textprops =\
        [{"color": color2[0],"fontweight": 'bold'},
        {"color": color[0],"fontweight": 'bold'}]
        
        fig_text(s = f'<{away}>' + ' ' + 'dominated the game with greater dominance' + '\n' + 'of the last third than their opponent' + ' ' + 
                 f'<{home}>.',
                 highlight_textprops = highlight_textprops,
                 x = 0.528, y = 0.88,
                 color='#181818', fontweight='bold',
                 ha='center', va='center',
                 fontsize=5)

    #############################################################################################################################################
    
    # Club Logo
    fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + club + '.png', fig=fig,
                    left=0.06, bottom=0.88, width=0.05, height=0.09)

    plt.savefig('assets/fieldTilt' + club + '.png', dpi=300)

    return app.get_asset_url('fieldTilt' + club + '.png')

################################################################################################################################################

def profilePlayer():
        
        fig, ax = plt.subplots(figsize=(15, 10))

        pitch = Pitch(pitch_type='opta',
                        pitch_color='#E8E8E8', line_color='#181818',
                        line_zorder=3, linewidth=0.5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        # Club Logo
        fig = add_image(image='Images/Clubs/' + 'Mundial' + '/' + 'Portugal' + '.png', fig=fig,
                        left=0.1, bottom=0.88, width=0.2, height=0.09)

        plt.savefig('assets/profile.png', dpi=300)

        return app.get_asset_url('profile.png')

################################################################################################################################################

def possessionGained(team, eventType):
    fig, ax = plt.subplots(figsize=(6, 4))

    pitch = Pitch(pitch_type='opta',
                    pitch_color='#E8E8E8', line_color='#181818',
                    line_zorder=3, linewidth=0.5, spot_scale=0.00)

    pitch.draw(ax=ax)

    fig.set_facecolor('#E8E8E8')

    defensiveActions = ['Aerial', 'Tackle', 'Foul', 'Interception', 'Clearance']

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
        [{"color": '#9a1534',"fontweight": 'bold'}]

    if eventType == 'BallRecovery':
        test = df.loc[(df['typedisplayName'] == 'BallRecovery') & (df['team'] == team)].reset_index(drop=True)

        fig_text(s = team + ' gained the most possession \n in their <defensive midfield>',
                        highlight_textprops = highlight_textprops,
                        x = 0.15, y = 0.895,
                        color='#181818',
                        fontsize=4)

    elif eventType == 'defensiveActions':
        test = df.loc[((df['typedisplayName'] == defensiveActions[0]) |
                      (df['typedisplayName'] == defensiveActions[1]) |
                      (df['typedisplayName'] == defensiveActions[2]) |
                      (df['typedisplayName'] == defensiveActions[3]) |
                      (df['typedisplayName'] == defensiveActions[4])) & (df['team'] == team)].reset_index(drop=True)

        fig_text(s = team + ' made the most defensive actions \n in their <defensive midfield>',
                        highlight_textprops = highlight_textprops,
                        x = 0.15, y = 0.895,
                        color='#181818',
                        fontsize=4)

    elif eventType == 'Pass':
        test = df.loc[(df['typedisplayName'] == 'Pass') & (df['team'] == team)].reset_index(drop=True)

        fig_text(s = team + ' made the most passes \n just before the <halfway line>',
                        highlight_textprops = highlight_textprops,
                        x = 0.15, y = 0.895,
                        color='#181818',
                        fontsize=4)


    elif eventType == 'ballLost':
        test = df.loc[(df['typedisplayName'] == 'Dispossessed') & (df['team'] == team)].reset_index(drop=True)

        fig_text(s = team + ' lost possession the most  \n just after the <halfway line>',
                        highlight_textprops = highlight_textprops,
                        x = 0.15, y = 0.895,
                        color='#181818',
                        fontsize=4)

        
    path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                    path_effects.Normal()]

    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                        ['#e8e8e8', '#9a1534'], N=10)

    bs = pitch.bin_statistic(test['x'], test['y'], statistic='count', bins=(6, 1), normalize=True)

    pitch.heatmap(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap)
            
    pitch.label_heatmap(bs, color='#E8E8E8', fontsize=12,
                                ax=ax, ha='center', va='bottom',
                                str_format='{:.0%}', path_effects=path_eff)

    fig_text(s = 'Possession Gained',
                    x = 0.5, y = 0.96,
                    color='#181818',
                    ha='center', va='center',
                    fontsize=14)

    fig_text(s = 'World Cup 2022',
                    x = 0.5, y = 0.91,
                    color='#181818', alpha=0.8,
                    ha='center', va='center',
                    fontsize=5)

    add_image(image='C:/Users/menes/Documents/Data Hub/Images/Clubs/' + 'Mundial' + '/' + team + '.png', fig=fig, left=0.25, bottom=0.905, width=0.08, height=0.09)
    
    add_image(image='C:/Users/menes/Documents/Data Hub/Images/WorldCup_Qatar.png', fig=fig, left=0.7, bottom=0.9, width=0.08, height=0.1)

    plt.savefig('assets/possessionGained' + team + '.png', dpi=300)

    return app.get_asset_url('possessionGained' + team + '.png')

################################################################################################################################################
#--------------------------------------------------- SCOUTING --------------------------------------------------------------------------------
################################################################################################################################################

"""
df = pd.read_csv('WyScout.csv')
"""

def playerAbility(df):

  def rank(df):
    for col in df.columns:
      df['rank_' + col] = df[col].rank(pct=True)

  rank(df)

  df['Pass Ability'] = ((df['rank_Passes/90'] * 65) + (df['rank_Accurate short / medium passes, %'] * 35))

  ##########################################################################################################################################################################

  df['KeyPass Ability'] = ((df['rank_Key passes/90'] * 65) + (df['rank_Smart passes/90'] * 35))

  ##########################################################################################################################################################################

  df['SetPieces Ability'] = ((df['rank_Corners/90']  * 60) + (df['rank_Direct free kicks on target, %'] * 40))

  ##########################################################################################################################################################################

  df['Dribbling Ability'] = ((df['rank_Dribbles/90']* 50) + (df['rank_Successful dribbles %'] * 50))

  ##########################################################################################################################################################################

  df['Create Chances Ability'] = ((df['rank_Shot assists/90'] * 10) + (df['rank_Second assists/90'] * 30) + (df['rank_Third assists/90'] * 15) +
                                      (df['rank_xA'] * 25) + (df['rank_Assists'] * 20))

  ##########################################################################################################################################################################

  df['Sight play'] = ((df['rank_Passes to penalty area/90'] * 20) + (df['rank_Key passes/90'] * 30) +
                            (df['rank_Passes final 1/3 %'] * 20) + (df['rank_Smart passes/90'] * 30))

  ##########################################################################################################################################################################

  df['Concentration Ability'] = ((df['rank_PAdj Sliding tackles'] * 50) + (df['rank_PAdj Interceptions'] * 50))

  ##########################################################################################################################################################################

  df['Finishing Ability'] = ((df['rank_Shots on target, %'] * 15) + (df['rank_Goals'] * 60) + (df['rank_xG'] * 25))

  ##########################################################################################################################################################################

  df['Heading Ability'] = ((df['rank_Head goals'] * 50) + (df['rank_Head goals/90'] * 50))

  ##########################################################################################################################################################################

  df['Interception Ability'] = ((df['rank_PAdj Interceptions'] * 65) + (df['Interceptions/90'] * 35))

  ##########################################################################################################################################################################

  df['Tackle Ability'] = ((df['rank_Sliding tackles/90'] * 35) + (df['rank_PAdj Sliding tackles'] * 65))

  ##########################################################################################################################################################################

  df['Aerial Ability'] = ((df['rank_Aerial duels %'] * 50) + (df['rank_Aerial duels/90'] * 50))

################################################################################################################################################

def radar_chart_compare(df, player, player2, cols):

  from soccerplots.radar_chart import Radar

  #Obtenção dos dois jogadores que pretendemos
  pl1 = df[(df['Player'] == player)]

  position = pl1['Position'].unique()
  position = position.tolist()
  position = position[0]
  if ', ' in position:
      position = position.split(', ')[0]

  val1 = pl1[cols].values[0]

  club = pl1['Team'].values[0]
  league = pl1['Comp'].values[0]

  #Obtenção dos dois jogadores que pretendemos
  pl2 = df[(df['Player'] == player2)]
  val2 = pl2[cols].values[0]

  position2 = pl2['Position'].unique()
  position2 = position2.tolist()
  position2 = position2[0]
  if ', ' in position2:
      position2 = position2.split(', ')[0]

  club2 = pl2['Team'].values[0]
  league2 = pl2['Comp'].values[0]

  #Obtenção dos valores das colunas que pretendemos colocar no radar chart, não precisamos aceder ao index porque só iriamos aceder aos valores de um dos jogadores
  values = [val1, val2]

  rango = df.loc[(df['Comp'] == league) & (df.Position.str.contains(position))].reset_index(drop=True)

  #Obtençaõ dos valores min e max das colunas selecionadas
  ranges = [(rango[col].min(), rango[col].max()) for col in cols] 

  #Atribuição dos valores aos titulos e respetivos tamanhos e cores
  title = dict(
      #Jogador 1
      title_name = player,
      title_color = '#548135',
      
      #Jogador 2
      title_name_2 = player2,
      title_color_2 = '#fb8c04',

      #Tamnhos gerais do radar chart
      title_fontsize = 20,
      subtitle_fontsize = 15,
    
      subtitle_name=club,
      subtitle_color='#181818',
      subtitle_name_2=club2

  )

  #team_player = df[col_name_team].to_list()

  #dict_team ={'Dortmund':['#ffe011', '#000000'],
              #'Nice':['#cc0000', '#000000'],
              #'Nice':['#cc0000', '#000000']}

  #color = dict_team.get(team_player[0])

  ## endnote 
  endnote = "Visualization made by: Pedro Meneses(@menesesp20)"

  #Criação do radar chart
  fig, ax = plt.subplots(figsize=(18,15), dpi=500)
  radar = Radar(background_color="#E8E8E8", patch_color="#181818", range_color="#181818", label_color="#181818", label_fontsize=10, range_fontsize=11)
  fig, ax = radar.plot_radar(ranges=ranges, 
                             params=cols, 
                             values=values, 
                             radar_color=['#548135','#fb8c04'], 
                             figax=(fig, ax),
                             title=title,
                             endnote=endnote, end_size=0, end_color="#1b1b1b",
                             compare=True)
  
  fig.set_facecolor('#E8E8E8')

################################################################################################################################################

def radar_chart(df, league, player, cols, player2=None):

  from soccerplots.radar_chart import Radar

  leagueDF = df.loc[df.Comp == league].reset_index(drop=True)

  tier = leagueDF.Tier.unique()
  tier = tier.tolist()
  tier = tier[0]

  if player2 == None:
    #Atribuição do jogador a colocar no gráfico
    players = df.loc[(df['Player'] == player) & (df['Season'] == '2021/22')].reset_index(drop=True)

    club = players.Team.unique()
    club = club.tolist()
    club = club[0]

    tierPlayer = players.Tier.unique()
    tierPlayer = tierPlayer.tolist()
    tierPlayer = tierPlayer[0]

    position = players['Position'].unique()
    position = position.tolist()
    position = position[0]
    if ', ' in position:
        position = position.split(', ')[0]

    """
    playersDF = []

    if (tier == 2) & (tierPlayer == 1):
      for i, row in players[Forward].iterrows():
          playersDF.append(row.values * 1.1)

      players = pd.DataFrame(playersDF)

      players.rename(columns={0: cols[0], 1: cols[1], 2: cols[2], 3: cols[3],
                                4: cols[4], 5: cols[5], 6: cols[6],
                                7: cols[7], 8: cols[8], 9: cols[9]}, inplace=True)

    #####################################################################################################################
    #####################################################################################################################

    elif (tier == 3) & (tierPlayer == 1):
      for i, row in players[Forward].iterrows():
          playersDF.append(row.values * 1.3)

      players = pd.DataFrame(playersDF)

      players.rename(columns={0: cols[0], 1: cols[1], 2: cols[2], 3: cols[3],
                                4: cols[4], 5: cols[5], 6: cols[6],
                                7: cols[7], 8: cols[8], 9: cols[9]}, inplace=True)

    elif (tier == 3) & (tierPlayer == 2):
      for i, row in players[Forward].iterrows():
          playersDF.append(row.values * 1.1)

      players = pd.DataFrame(playersDF)

      players.rename(columns={0: cols[0], 1: cols[1], 2: cols[2], 3: cols[3],
                                4: cols[4], 5: cols[5], 6: cols[6],
                                7: cols[7], 8: cols[8], 9: cols[9]}, inplace=True)

    #####################################################################################################################
    #####################################################################################################################

    elif (tier == 4) & (tierPlayer == 1):
      for i, row in players[Forward].iterrows():
          playersDF.append(row.values * 1.35)

      players = pd.DataFrame(playersDF)

      players.rename(columns={0: cols[0], 1: cols[1], 2: cols[2], 3: cols[3],
                                4: cols[4], 5: cols[5], 6: cols[6],
                                7: cols[7], 8: cols[8], 9: cols[9]}, inplace=True)

    elif (tier == 4) & (tierPlayer == 2):
      for i, row in players[Forward].iterrows():
          playersDF.append(row.values * 1.25)

      players = pd.DataFrame(playersDF)

      players.rename(columns={0: cols[0], 1: cols[1], 2: cols[2], 3: cols[3],
                                4: cols[4], 5: cols[5], 6: cols[6],
                                7: cols[7], 8: cols[8], 9: cols[9]}, inplace=True)

    elif (tier == 4) & (tierPlayer == 3):
      for i, row in players[Forward].iterrows():
          playersDF.append(row.values * 1.15)

      players = pd.DataFrame(playersDF)

      players.rename(columns={0: cols[0], 1: cols[1], 2: cols[2], 3: cols[3],
                                4: cols[4], 5: cols[5], 6: cols[6],
                                7: cols[7], 8: cols[8], 9: cols[9]}, inplace=True)
    """
      

    #####################################################################################################################
    #####################################################################################################################

    #Valores que pretendemos visualizar no radar chart, acedemos ao index 0 para obtermos os valores dentro da lista correta
    values = players[cols].values[0]
    #Obtenção do alcance minimo e máximo dos valores

    rango = df.loc[(df['Comp'] == league) & (df.Position.str.contains(position)) | (df.Player == player)].reset_index(drop=True)

    ranges = [(rango[col].min(), rango[col].max()) for col in cols]

    color = ['#548135','#fb8c04']
    #Atribuição dos valores aos titulos e respetivos tamanhos e cores
    title = dict(
      title_name = player,
      title_color = color[0],
      title_fontsize = 25,
      subtitle_fontsize = 15,
    
      subtitle_name=club,
      subtitle_color='#181818',
    )

    #team_player = df[col_name_team].to_list()

    #dict_team ={'Dortmund':['#ffe011', '#000000'],
              #'Nice':['#cc0000', '#000000'],}

    #color = dict_team.get(team_player[0])

    ## endnote 
    endnote = "Visualization made by: Pedro Meneses(@menesesp20)"

    #Criação do radar chart
    fig, ax = plt.subplots(figsize=(18,15))
    radar = Radar(background_color="#E8E8E8", patch_color="#181818", range_color="#181818", label_color="#181818", label_fontsize=10, range_fontsize=11)
    fig, ax = radar.plot_radar(ranges=ranges, 
                               params=cols, 
                               values=values, 
                               radar_color=color,
                               figax=(fig, ax),
                               image_coord=[0.464, 0.81, 0.1, 0.075],
                               title=title,
                               endnote=endnote)

    fig.set_facecolor('#E8E8E8')

  else:
    radar_chart_compare(df, player, player2, 'Player', 'Team', cols, league)

################################################################################################################################################

def PCA5(df, age, end_Value, minutes, playerName, role, contract, Tier, tierSec=None):

    if (age == 'All Data') & (Tier == 'All Data') & (contract == 'No'):
        # Age and League == All Data - DONE - 8
        df = df.loc[(df['Minutes played'] >= minutes) & (df['Market value'] >= 0) & (df['Market value'] <= end_Value)].reset_index(drop=True)

    elif (age == 'All Data') & (Tier == 'All Data') & (contract == 'Yes'):
        # Age and League == All Data - DONE - 8
        df = df.loc[(df['Minutes played'] >= minutes) & (df['Market value'] >= 0) & (df['Market value'] <= end_Value) &
                    (df['Contract expires'] == str(date.today().year) + '-06' + '-30')].reset_index(drop=True)

####################################################################################################################################################################
####################################################################################################################################################################


    elif (age != 'All Data') & (Tier != 'All Data') & (contract == 'No') & (tierSec == None):
        # Age and Leagues != All Data and Leagues by Tier - DONE - 8
        df = df.loc[(df.Age <= age) & (df.Tier == Tier) & (df['Minutes played'] >= minutes) &
                    (df['Market value'] >= 0) & (df['Market value'] <= end_Value) | (df.Player == playerName)].reset_index(drop=True)

    elif (age != 'All Data') & (Tier != 'All Data') & (contract == 'Yes') & (tierSec == None):
        # Age and Leagues != All Data and Leagues by Tier - DONE - 8
        df = df.loc[(df.Age <= age) & (df.Tier == Tier) & (df['Minutes played'] >= minutes) &
                    (df['Contract expires'] == str(date.today().year) + '-06' + '-30') &
                    (df['Market value'] >= 0) & (df['Market value'] <= end_Value) | (df.Player == playerName)].reset_index(drop=True)

    elif (age != 'All Data') & (Tier != 'All Data') & (tierSec != None) & (contract == 'No'):
        # Age and Leagues != All Data and Leagues by Tier - DONE - 8
        df = df.loc[(df.Age <= age) & (df['Minutes played'] >= minutes) &
                    (df['Market value'] >= 0) & (df['Market value'] <= end_Value) &
                    ((df.Tier == Tier) | (df.Tier == tierSec)) | (df.Player == playerName)].reset_index(drop=True)

    elif (age != 'All Data') & (Tier != 'All Data') & (tierSec != None) & (contract == 'Yes'):
        # Age and Leagues != All Data and Leagues by Tier - DONE - 8
        df = df.loc[(df.Age <= age) & (df['Minutes played'] >= minutes) &
                    (df['Contract expires'] == str(date.today().year) + '-06' + '-30') &
                    (df['Market value'] >= 0) & (df['Market value'] <= end_Value) &
                    ((df.Tier == Tier) | (df.Tier == tierSec)) | (df.Player == playerName)].reset_index(drop=True)

####################################################################################################################################################################
####################################################################################################################################################################

    elif (age != 'All Data') & (Tier == 'All Data') & (contract == 'No'):
        # League == All Data - DONE - 8
        df = df.loc[(df.Age <= age) & (df['Minutes played'] >= minutes) &
                        (df['Market value'] >= 0) & (df['Market value'] <= end_Value) |
                        (df.Player == playerName)].reset_index(drop=True)

    elif (age != 'All Data') & (Tier == 'All Data') & (contract == 'Yes'):
        # League == All Data - DONE - 8
        df = df.loc[(df.Age <= age) & (df['Minutes played'] >= minutes) &
                    (df['Contract expires'] == str(date.today().year) + '-06' + '-30') &
                    (df['Market value'] >= 0) & (df['Market value'] <= end_Value) |
                    (df.Player == playerName)].reset_index(drop=True)

####################################################################################################################################################################
####################################################################################################################################################################

    elif (age == 'All Data') & (Tier != 'All Data') & (contract == 'No'):
        # Age == All Data - DONE - 4
        df = df.loc[(df['Minutes played'] >= minutes) & (df['Market value'] >= 0) & (df['Market value'] <= end_Value) &
                    ((df.Tier == Tier) | (df.Tier == tierSec)) | (df.Player == playerName)].reset_index(drop=True)

    elif (age == 'All Data') & (Tier != 'All Data') & (contract == 'Yes'):
        # Age == All Data - DONE - 4
        df = df.loc[(df['Minutes played'] >= minutes) & (df['Market value'] >= 0) & (df['Contract expires'] == str(date.today().year) + '-06' + '-30') &
                    (df['Market value'] <= end_Value) & ((df.Tier == Tier) | (df.Tier == tierSec)) | (df.Player == playerName)].reset_index(drop=True)
                        
#######################################################################################################################################################################################
#######################################################################################################################################################################################

    if role == center_Back:
        df = df.loc[(df.Position.str.contains('CB'))][['Player', 'Non-penalty goals/90', 'Offensive duels %', 'Progressive runs/90',
                'Passes %', 'Forward passes %', 'Forward passes/90', 'Progressive passes/90',
                'PAdj Interceptions', 'PAdj Sliding tackles', 'Defensive duels/90', 'Defensive duels %',
                'Aerial duels/90', 'Aerial duels %', 'Shots blocked/90']]

    elif role == full_Back:
        df = df.loc[(df.Position.str.contains('RB')) | (df.Position.str.contains('LB'))][['Player', 'Successful dribbles %', 'Touches in box/90', 'Offensive duels %', 'Progressive runs/90', 'Crosses/90', 'Deep completed crosses/90',
            'Passes %', 'Deep completions/90', 'Progressive passes/90', 'Key passes/90', 'Third assists/90',
                'PAdj Interceptions', 'Defensive duels %', 'Aerial duels/90', 'Aerial duels %']]

    elif role == defensive_Midfield:

        df = df.loc[(df.Position.str.contains('RCMF')) | (df.Position.str.contains('LCMF')) | (df.Position.str.contains('LDMF')) | (df.Position.str.contains('RDMF')) |
                    (df.Position.str.contains('AMF')) | (df.Position.str.contains('DMF'))][['Player', 'xG/90', 'Shots', 'Progressive runs/90', 'Successful dribbles %',
                        'Passes %', 'Forward passes %', 'Forward passes/90', 'Progressive passes/90','PAdj Sliding tackles',
                        'PAdj Interceptions', 'Aerial duels %', 'Defensive duels %', 'Offensive duels %']]

    elif role == Midfield:

        df = df.loc[(df.Position.str.contains('RCMF')) | (df.Position.str.contains('LCMF')) | (df.Position.str.contains('LDMF')) | (df.Position.str.contains('RDMF')) |
                    (df.Position.str.contains('AMF')) | (df.Position.str.contains('DMF'))][['Player', 'xG/90', 'Shots', 'Progressive runs/90', 'Successful dribbles %',
                'Passes %', 'Forward passes %', 'Forward passes/90', 'Progressive passes/90',
                'Key passes/90', 'Second assists/90', 'Assists', 'xA',
                'PAdj Interceptions', 'Aerial duels %', 'Defensive duels %']]

    elif role == offensive_Midfield:

        df = df.loc[(df.Position.str.contains('RCMF')) | (df.Position.str.contains('LCMF')) | (df.Position.str.contains('LDMF')) | (df.Position.str.contains('RDMF')) |
                    (df.Position.str.contains('AMF')) | (df.Position.str.contains('DMF'))][['Player', 'xG/90', 'Goals/90', 'Progressive runs/90', 'Successful dribbles %',
                        'xA/90', 'Deep completions/90', 'Passes to penalty area/90',
                        'Touches in box/90', 'Key passes/90', 'Passes final 1/3 %',
                        'Passes penalty area %', 'Progressive passes/90',
                        'Succ defensive actions/90', 'PAdj Interceptions', 'Aerial duels %', 'Defensive duels %']]

    elif role == Winger:

        df = df.loc[(df.Position.str.contains('LW')) | (df.Position.str.contains('RW')) | (df.Position.str.contains('LAMF')) |
                     (df.Position.str.contains('RAMF')) | (df.Position.str.contains('LWF')) | (df.Position.str.contains('RWF'))][['Player', 'Successful dribbles %', 'Goals', 'xG/90',
            'xA/90', 'Touches in box/90', 'Dribbles/90', 'Passes to penalty area/90', 'Key passes/90',
            'Progressive runs/90', 'Crosses/90', 'Deep completed crosses/90',
            'Offensive duels/90', 'PAdj Interceptions']]

    elif role == Forward:

        df = df.loc[(df.Position.str.contains('CF'))][['Player', 'Goals', 'xG/90', 'Shots on target, %', 'Goal conversion, %',
            'Successful dribbles %', 'xA/90', 'Touches in box/90', 'Dribbles/90',
            'Offensive duels/90', 'PAdj Interceptions', 'Aerial duels/90', 'Aerial duels %']]

    else:

        df = df

    #Guardar como matriz na variavél X todas as métricas e na variavél y todos os nomes dos jogadores
    X, y = df.iloc[:, 1:len(df.columns)].values, df.iloc[:, 0].values

    #Escalar os dados, passo muito importante em Machine Learning
    X_std = StandardScaler().fit_transform(X)

    #Aplicar o método PCA, ou seja reduzir o dataframe até à quantidade necessária de dados, sem perdermos informação essencial
    if (age == 'All Data') & (Tier != 'All Data'):
        pca = PCA(n_components = 10)
        pca.fit(X_std)
        X_pca = pca.transform(X_std)
        #Visualização de quantas dimensões nos interessam
        #print("Shape x_PCA: ", X_pca.shape)
        #expl = pca.explained_variance_ratio_

        #for x in range(0, len(df.columns), 2):
        #    print("Explained Variance: " + str(x) + " components:", sum(expl[0:x]))

        #plt.plot(np.cumsum(pca.explained_variance_ratio_))
        #plt.xlabel('Dimensions')
        #plt.ylabel('Explained Variance')

        #Utilizar n dimensões que nos indicar o Elbow
        N_COMP = 4

    elif len(df) < 10:

        pca = PCA(n_components = len(df))
        pca.fit(X_std)
        X_pca = pca.transform(X_std)
        #Visualização de quantas dimensões nos interessam
        #print("Shape x_PCA: ", X_pca.shape)
        #expl = pca.explained_variance_ratio_

        #for x in range(0, len(df.columns), 2):
        #    print("Explained Variance: " + str(x) + " components:", sum(expl[0:x]))

        #plt.plot(np.cumsum(pca.explained_variance_ratio_))
        #plt.xlabel('Dimensions')
        #plt.ylabel('Explained Variance')

        #Utilizar n dimensões que nos indicar o Elbow
        N_COMP = 4
        
    else:

        pca = PCA(n_components = len(df.columns)-1)
        pca.fit(X_std)
        X_pca = pca.transform(X_std)
        #Visualização de quantas dimensões nos interessam
        #print("Shape x_PCA: ", X_pca.shape)
        #expl = pca.explained_variance_ratio_

        #for x in range(0, len(df.columns), 2):
        #    print("Explained Variance: " + str(x) + " components:", sum(expl[0:x]))

        #plt.plot(np.cumsum(pca.explained_variance_ratio_))
        #plt.xlabel('Dimensions')
        #plt.ylabel('Explained Variance')

        #Utilizar n dimensões que nos indicar o Elbow
        N_COMP = 8

    columns = []

    for col in range(1, N_COMP+1):
        columns.append("PCA" + str(col))

    df_pca_resultado = pd.DataFrame(data=X_pca[:,:N_COMP], columns=columns, index = y)

    #Mostrar a correlação numa matriz
    corr_matrix = df_pca_resultado.T.corr(method='pearson')

    #Função automatizada para encontrarmos o jogador que pretendemos e o jogadores mais similar a ele
    def GetSimilarPlayers(playerName, numPlayers, corr_matrix):
        
        SimPlayers = pd.DataFrame(columns = ['PlayerName', 'Similar Player', 'Correlation Factor'])

        i = 0
        for i in range(0, numPlayers):
            row = corr_matrix.loc[corr_matrix.index == playerName].squeeze()

            SimPlayers.at[i, 'PlayerName'] = playerName
            SimPlayers.at[i, 'Similar Player'] = row.nlargest(i+2).sort_values(ascending=True).index[0]
            SimPlayers.at[i, 'Correlation Factor'] = row.nlargest(i+2).round(2).sort_values(ascending=True)[0]

            i = i+1
        
        return SimPlayers
    
    PlayerName = playerName
    NumPlayers = 5

    df_correlatedPlayers = GetSimilarPlayers(playerName, NumPlayers, corr_matrix)

    df_correlatedPlayers.drop_duplicates(inplace=True)

    return df_correlatedPlayers

################################################################################################################################################

def similarityDashboard(df, age, end_Value, minutes, playerName, role, contract, Tier, tierSec=None):

        from soccerplots.radar_chart import Radar

        df1 = PCA5(df, age, end_Value, minutes, playerName, role, contract, Tier, tierSec)

        #df1 = PCA5(wyscout, 25, 250000, 1000, 'Vinícius Júnior', Winger, 'No', 0, 3)

        fig = plt.figure(figsize=(50, 45), dpi = 500, facecolor = '#E8E8E8')
        gspec = gridspec.GridSpec(
        ncols=7, nrows=2, wspace = 0.5
        )

        ########################################################################################################################################################

        ax6 = plt.subplot(
                        gspec[1, 0],
                )

        dfPlayer = df.loc[df.Player == playerName].reset_index(drop=True)

        position = dfPlayer.Position.unique()
        position = position.tolist()
        position = position[0]
        if ', ' in position:
                position = position.split(', ')[0]

        rows = 9
        cols = 0

        #Criação da lista de jogadores
        Players = df1['PlayerName'].unique()

        Players = Players.tolist()

        #Criação da lista de jogadores similares
        similarPlayers = df1['Similar Player'].unique()

        similarPlayers = similarPlayers.tolist()


        pl1 = df.loc[df['Player'] == df1['Similar Player'].iloc[0]]
        pl2 = df.loc[df['Player'] == df1['Similar Player'].iloc[1]]
        pl3 = df.loc[df['Player'] == df1['Similar Player'].iloc[2]]
        pl4 = df.loc[df['Player'] == df1['Similar Player'].iloc[3]]
        pl5 = df.loc[df['Player'] == df1['Similar Player'].iloc[4]]

        team = []

        team.append(pl1['Team'].values)
        team.append(pl2['Team'].values)
        team.append(pl3['Team'].values)
        team.append(pl4['Team'].values)
        team.append(pl5['Team'].values)

        comp = []

        comp.append(pl1['Comp'].values)
        comp.append(pl2['Comp'].values)
        comp.append(pl3['Comp'].values)
        comp.append(pl4['Comp'].values)
        comp.append(pl5['Comp'].values)

        xG90 = []

        xG90.append(pl1['xG/90'].values)
        xG90.append(pl2['xG/90'].values)
        xG90.append(pl3['xG/90'].values)
        xG90.append(pl4['xG/90'].values)
        xG90.append(pl5['xG/90'].values)

        Goals90 = []

        Goals90.append(pl1['Goals/90'].values)
        Goals90.append(pl2['Goals/90'].values)
        Goals90.append(pl3['Goals/90'].values)
        Goals90.append(pl4['Goals/90'].values)
        Goals90.append(pl5['Goals/90'].values)
        
        Progressive_runs90 = []

        Progressive_runs90.append(pl1['Progressive runs/90'].values)
        Progressive_runs90.append(pl2['Progressive runs/90'].values)
        Progressive_runs90.append(pl3['Progressive runs/90'].values)
        Progressive_runs90.append(pl4['Progressive runs/90'].values)
        Progressive_runs90.append(pl5['Progressive runs/90'].values)
        
        dribbles = []

        dribbles.append(pl1['Successful dribbles %'].values)
        dribbles.append(pl2['Successful dribbles %'].values)
        dribbles.append(pl3['Successful dribbles %'].values)
        dribbles.append(pl4['Successful dribbles %'].values)
        dribbles.append(pl5['Successful dribbles %'].values)
        
        xA = []

        xA.append(pl1['xA/90'].values)
        xA.append(pl2['xA/90'].values)
        xA.append(pl3['xA/90'].values)
        xA.append(pl4['xA/90'].values)
        xA.append(pl5['xA/90'].values)
        
        Deepcompletions = []

        Deepcompletions.append(pl1['Deep completions/90'].values)
        Deepcompletions.append(pl2['Deep completions/90'].values)
        Deepcompletions.append(pl3['Deep completions/90'].values)
        Deepcompletions.append(pl4['Deep completions/90'].values)
        Deepcompletions.append(pl5['Deep completions/90'].values)
        
        Penaltyarea90 = []

        Penaltyarea90.append(pl1['Passes to penalty area/90'].values)
        Penaltyarea90.append(pl2['Passes to penalty area/90'].values)
        Penaltyarea90.append(pl3['Passes to penalty area/90'].values)
        Penaltyarea90.append(pl4['Passes to penalty area/90'].values)
        Penaltyarea90.append(pl5['Passes to penalty area/90'].values)

        
        Market_value = []

        Market_value.append(pl1['Market value'].values)
        Market_value.append(pl2['Market value'].values)
        Market_value.append(pl3['Market value'].values)
        Market_value.append(pl4['Market value'].values)
        Market_value.append(pl5['Market value'].values)
        
        keyPasses = []

        keyPasses.append(pl1['Key passes/90'].values)
        keyPasses.append(pl2['Key passes/90'].values)
        keyPasses.append(pl3['Key passes/90'].values)
        keyPasses.append(pl4['Key passes/90'].values)
        keyPasses.append(pl5['Key passes/90'].values)

        passesFinaThird = []

        passesFinaThird.append(pl1['Passes final 1/3 %'].values)
        passesFinaThird.append(pl2['Passes final 1/3 %'].values)
        passesFinaThird.append(pl3['Passes final 1/3 %'].values)
        passesFinaThird.append(pl4['Passes final 1/3 %'].values)
        passesFinaThird.append(pl5['Passes final 1/3 %'].values)

        keyPenBox = []

        keyPenBox.append(pl1['Passes penalty area %'].values)
        keyPenBox.append(pl2['Passes penalty area %'].values)
        keyPenBox.append(pl3['Passes penalty area %'].values)
        keyPenBox.append(pl4['Passes penalty area %'].values)
        keyPenBox.append(pl5['Passes penalty area %'].values)

        progressivePasses = []

        progressivePasses.append(pl1['Progressive passes/90'].values)
        progressivePasses.append(pl2['Progressive passes/90'].values)
        progressivePasses.append(pl3['Progressive passes/90'].values)
        progressivePasses.append(pl4['Progressive passes/90'].values)
        progressivePasses.append(pl5['Progressive passes/90'].values)

        succDefensiveActions = []

        succDefensiveActions.append(pl1['Succ defensive actions/90'].values)
        succDefensiveActions.append(pl2['Succ defensive actions/90'].values)
        succDefensiveActions.append(pl3['Succ defensive actions/90'].values)
        succDefensiveActions.append(pl4['Succ defensive actions/90'].values)
        succDefensiveActions.append(pl5['Succ defensive actions/90'].values)

        PAdjInterceptions = []

        PAdjInterceptions.append(pl1['PAdj Interceptions'].values)
        PAdjInterceptions.append(pl2['PAdj Interceptions'].values)
        PAdjInterceptions.append(pl3['PAdj Interceptions'].values)
        PAdjInterceptions.append(pl4['PAdj Interceptions'].values)
        PAdjInterceptions.append(pl5['PAdj Interceptions'].values)

        aerialDuels = []

        aerialDuels.append(pl1['Aerial duels %'].values)
        aerialDuels.append(pl2['Aerial duels %'].values)
        aerialDuels.append(pl3['Aerial duels %'].values)
        aerialDuels.append(pl4['Aerial duels %'].values)
        aerialDuels.append(pl5['Aerial duels %'].values)

        defensiveDuels = []

        defensiveDuels.append(pl1['Defensive duels %'].values)
        defensiveDuels.append(pl2['Defensive duels %'].values)
        defensiveDuels.append(pl3['Defensive duels %'].values)
        defensiveDuels.append(pl4['Defensive duels %'].values)
        defensiveDuels.append(pl5['Defensive duels %'].values)
        #Valores de similariedade
        valuePlayers = df1['Correlation Factor']
        valuePlayers = valuePlayers.tolist()

        data = {
                'Players' : similarPlayers,
                'Similarity' : valuePlayers,
                'Market value' : Market_value,
                'xG/90' : xG90,
                'Goals/90' : Goals90,
                'Progressive runs/90' : Progressive_runs90,
                'Succ dribbles %' : dribbles,
                'xA/90' : xA,
                'Deep completions/90' : Deepcompletions,
                'Passes to penalty area/90' : Penaltyarea90,
                'Key passes/90' : keyPasses,
                'Passes final 1/3 %' : passesFinaThird,
                'Passes penalty area %' : keyPenBox,
                'Progressive passes/90' : progressivePasses,
                'Succ defensive actions/90' : succDefensiveActions,
                'PAdj Interceptions' : PAdjInterceptions,
                'Aerial duels %' : aerialDuels,
                'Defensive duels %' : defensiveDuels

        }

        data = pd.DataFrame(data)

        data.reset_index(drop=True, inplace=True)

        data = data.T

        data = data.to_dict('index')

        for i in range(1):
                for k, v in data.items():
                        if k == 'Players':
                                ax6.text(x=8, y=11, s=v[0], va='center', ha='center', weight='bold', size=48, color='#548135')
                                ax6.text(x=11.5, y=11, s=v[1], va='center', ha='center', weight='bold', size=48, color='#548135')
                                ax6.text(x=14.8, y=11, s=v[2], va='center', ha='center', weight='bold', size=48, color='#548135')
                                ax6.text(x=17.5, y=11, s=v[3], va='center', ha='center', weight='bold', size=48, color='#548135')
                                ax6.text(x=20.8, y=11, s=v[4], va='center', ha='center', weight='bold', size=48, color='#548135')
                                # shots column - this is my "main" column, hence bold text

                        if k == 'Similarity':
                                ax6.text(x=8, y=10.3, s=v[0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=10.3, s=v[1], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=10.3, s=v[2], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=10.3, s=v[3], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=10.3, s=v[4], va='center', ha='right', size=48, color='#181818')

                        if k == 'Market value':
                                ax6.text(x=8, y=9.6, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=9.6, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=9.6, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=9.6, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=9.6, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Progressive runs/90':
                                ax6.text(x=8, y=8.9, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=8.9, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=8.9, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=8.9, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=8.9, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Succ dribbles %':
                                ax6.text(x=8, y=8.2, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=8.2, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=8.2, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=8.2, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=8.2, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'xA/90':
                                ax6.text(x=8, y=7.5, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=7.5, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=7.5, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=7.5, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=7.5, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Deep completions/90':
                                ax6.text(x=8, y=6.8, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=6.8, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=6.8, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=6.8, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=6.8, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Passes to penalty area/90':
                                ax6.text(x=8, y=6.1, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=6.1, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=6.1, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=6.1, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=6.1, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'xG/90':
                                ax6.text(x=8, y=5.4, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=5.4, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=5.4, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=5.4, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=5.4, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Key passes/90':
                                ax6.text(x=8, y=4.7, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=4.7, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=4.7, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=4.7, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=4.7, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Passes final 1/3 %':
                                ax6.text(x=8, y=4, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=4, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=4, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=4, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=4, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Passes penalty area %':
                                ax6.text(x=8, y=3.3, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=3.3, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=3.3, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=3.3, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=3.3, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Progressive passes/90':
                                ax6.text(x=8, y=2.6, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=2.6, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=2.6, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=2.6, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=2.6, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Succ defensive actions/90':
                                ax6.text(x=8, y=1.9, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=1.9, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=1.9, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=1.9, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=1.9, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'PAdj Interceptions':
                                ax6.text(x=8, y=1.2, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=1.2, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=1.2, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=1.2, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=1.2, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Aerial duels %':
                                ax6.text(x=8, y=0.5, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=0.5, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=0.5, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=0.5, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=0.5, s=v[4][0], va='center', ha='right', size=48, color='#181818')

                        if k == 'Defensive duels %':
                                ax6.text(x=8, y=-0.2, s=v[0][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=11.5, y=-0.2, s=v[1][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=14.8, y=-0.2, s=v[2][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=17.5, y=-0.2, s=v[3][0], va='center', ha='right', size=48, color='#181818')
                                ax6.text(x=20.8, y=-0.2, s=v[4][0], va='center', ha='right', size=48, color='#181818')


        ax6.text(1, 11, 'Players', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 10.3, 'Similarity', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 9.6, 'Market value', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 8.9, 'Progressive runs/90', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 8.2, 'Succ Dribbles %', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 7.5, 'xA/90', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 6.8, 'Deep Completions/90', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 6.1, 'Passes penalty area/90', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 5.4, 'xG/90', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 4.7, 'Key passes/90', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 4, 'Passes final 1/3 %', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 3.3, 'Passes penalty area %', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 2.6, 'Progressive passes/90', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 1.9, 'Succ defensive actions/90', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 1.2, 'PAdj Interceptions', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, 0.5, 'Aerial duels %', weight='bold', ha='left', size=50, color='#548135')

        ax6.text(1, -0.2, 'Defensive duels %', weight='bold', ha='left', size=50, color='#548135')

        for row in range(rows):
                ax6.plot(
                [0, cols + 2],
                [row -.5, row - .5],
                ls=':',
                lw='.5',
                c='#e8e8e8'
                )

        ax6.axis('off')

        """
        for i in range(len(data)):

                ax6_image = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + 'Premier League' + '/' + data[0].get('Players') + '.png', fig=fig, left=0.09, bottom=0.16, width=0.1, height=0.058)

                ax6_image = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + 'Premier League' + '/' + data[1].get('Players') + '.png', fig=fig, left=0.09, bottom=0.242, width=0.1, height=0.058)

                ax6_image = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + 'Premier League' + '/' + data[2].get('Players') + '.png', fig=fig, left=0.09, bottom=0.322, width=0.1, height=0.058)

                ax6_image = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + 'Premier League' + '/' + data[3].get('Players')+ '.png', fig=fig, left=0.09, bottom=0.398, width=0.1, height=0.058)

                ax6_image = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + 'Premier League' + '/' + data[4].get('Players') + '.png', fig=fig, left=0.09, bottom=0.47, width=0.1, height=0.058)

                ax6_image = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + 'Premier League' + '/' + data[5].get('Players') + '.png', fig=fig, left=0.09, bottom=0.55, width=0.1, height=0.058)

                ax6_image = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + 'Premier League' + '/' + data[6].get('Players') + '.png', fig=fig, left=0.09, bottom=0.624, width=0.1, height=0.058)

                ax6_image = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + 'Premier League' + '/' + data[7].get('Players') + '.png', fig=fig, left=0.09, bottom=0.70, width=0.1, height=0.058)

                ax6_image = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + 'Premier League' + '/' + data[8].get('Players') + '.png', fig=fig, left=0.09, bottom=0.773, width=0.1, height=0.058)

                ax6_image = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + 'Premier League' + '/' + data[9].get('Players') + '.png', fig=fig, left=0.09, bottom=0.85, width=0.1, height=0.058)
        """

        ax1 = plt.subplot(
                        gspec[0, 1],
                )

        cols = role

        #PLAYER 1
        pl1 = df.loc[df['Player'] == playerName]
        values = pl1[cols].values[0]

        #Obtenção do alcance minimo e máximo dos valores
        ranges = [(df[col].min(), df[col].max()) for col in cols]

        #Criação do radar chart
        radar = Radar(background_color="#181818", patch_color='#181818', range_color="#181818", label_color="#181818", label_fontsize=0, range_fontsize=0)
        fig, ax1 = radar.plot_radar(ranges=ranges, 
                                params=cols, 
                                values=values, 
                                radar_color=['#548135', '#548135'],
                                figax=(fig, ax1),
                                end_size=0, end_color="#1b1b1b")

        age = pl1['Age'].unique()
        age = age.tolist()
        age = age[0]

        tier = pl1['Tier'].unique()
        tier = tier.tolist()
        tier = tier[0]
 
        ax_text(x = -20, y = -25,
                s=playerName,
                size=48,
                color='#181818',
                ax=ax1)

        ax_text(x = -16, y = -32,
                s='(' + str(age) + ')',
                size=40,
                color='#181818',
                ax=ax1)


        team = pl1['Team'].unique()
        team = team.tolist()
        team = team[0]

        ax_text(x = -5, y = -19,
                s=team + ' ' + '(' + str(tier) + ')',
                size=35,
                color='#181818',
                ax=ax1)

        ########################################################################################################################################################

        ax2 = plt.subplot(
                        gspec[0, 2],
                )

        cols = role

        #PLAYER 1
        pl1 = df.loc[df['Player'] == data.get('Players')[0]]
        val1 = pl1[cols].values[0]

        # PLAYER 2
        pl2 = df.loc[df['Player'] == playerName]
        val2 = pl2[cols].values[0]

        values = [val1, val2]

        #Obtenção do alcance minimo e máximo dos valores
        ranges = [(df[col].min(), df[col].max()) for col in cols]

        #Criação do radar chart
        radar = Radar(background_color="#181818", patch_color='#181818', range_color="#181818", label_color="#181818", label_fontsize=0, range_fontsize=0)
        fig, ax2 = radar.plot_radar(ranges=ranges, 
                                params=cols, 
                                values=values, 
                                radar_color=['#2d92df', '#548135'],
                                figax=(fig, ax2),
                                end_size=0, end_color="#1b1b1b",
                                compare=True)

        age = pl1['Age'].unique()
        age = age.tolist()
        age = age[0]

        tier = pl1['Tier'].unique()
        tier = tier.tolist()
        tier = tier[0]

        ax_text(x = -8, y = -27,
                s=data.get('Players')[0],
                size=48,
                color='#181818',
                ax=ax2)

        ax_text(x = -4, y = -34,
                s='(' + str(age) + ')',
                size=40,
                color='#181818',
                ax=ax2)

        team = pl1['Team'].unique()
        team = team.tolist()
        team = team[0]

        ax_text(x = -5, y = -19,
                s=team + ' ' + '(' + str(tier) + ')',
                size=35,
                color='#181818',
                ax=ax2)

        ########################################################################################################################################################    

        ax3 = plt.subplot(
                        gspec[0, 3],
                )

        cols = role

        #PLAYER 1
        pl1 = df.loc[df['Player'] == data.get('Players')[1]]
        val1 = pl1[cols].values[0]

        # PLAYER 2
        pl2 = df.loc[df['Player'] == playerName]
        val2 = pl2[cols].values[0]

        values = [val1, val2]
        #Obtenção do alcance minimo e máximo dos valores
        ranges = [(df[col].min(), df[col].max()) for col in cols]

        #Criação do radar chart
        radar = Radar(background_color="#181818", patch_color='#181818', range_color="#181818", label_color="#181818", label_fontsize=0, range_fontsize=0)
        fig, ax3 = radar.plot_radar(ranges=ranges, 
                                params=cols, 
                                values=values, 
                                radar_color=['#2d92df', '#548135'],
                                figax=(fig, ax3),
                                end_size=0, end_color="#1b1b1b",
                                compare=True)

        age = pl1['Age'].unique()
        age = age.tolist()
        age = age[0]

        tier = pl1['Tier'].unique()
        tier = tier.tolist()
        tier = tier[0]

        ax_text(x = -8, y = -27,
                s=data.get('Players')[1],
                size=48,
                color='#181818',
                ax=ax3)

        ax_text(x = -4, y = -34,
                s='(' + str(age) + ')',
                size=40,
                color='#181818',
                ax=ax3)

        team = pl1['Team'].unique()
        team = team.tolist()
        team = team[0]

        ax_text(x = -5, y = -19,
                s=team + ' ' + '(' + str(tier) + ')',
                size=35,
                color='#181818',
                ax=ax3)

        ########################################################################################################################################################

        ax4 = plt.subplot(
                        gspec[0, 4],
                )

        cols = role

        #PLAYER 1
        pl1 = df.loc[df['Player'] == data.get('Players')[2]]
        val1 = pl1[cols].values[0]

        # PLAYER 2
        pl2 = df.loc[df['Player'] == playerName]
        val2 = pl2[cols].values[0]

        values = [val1, val2]
        #Obtenção do alcance minimo e máximo dos valores
        ranges = [(df[col].min(), df[col].max()) for col in cols]

        #Criação do radar chart
        radar = Radar(background_color="#181818", patch_color='#181818', range_color="#181818", label_color="#181818", label_fontsize=0, range_fontsize=0)
        fig, ax4 = radar.plot_radar(ranges=ranges, 
                                params=cols, 
                                values=values, 
                                radar_color=['#2d92df', '#548135'],
                                figax=(fig, ax4),
                                end_size=0, end_color="#181818",
                                compare=True)

        age = pl1['Age'].unique()
        age = age.tolist()
        age = age[0]

        tier = pl1['Tier'].unique()
        tier = tier.tolist()
        tier = tier[0]

        ax_text(x = -8, y = -27,
                s=data.get('Players')[2],
                size=48,
                color='#181818',
                ax=ax4)

        ax_text(x = -4, y = -34,
                s='(' + str(age) + ')',
                size=40,
                color='#181818',
                ax=ax4)

        team = pl1['Team'].unique()
        team = team.tolist()
        team = team[0]

        ax_text(x = -5, y = -19,
                s=team + ' ' + '(' + str(tier) + ')',
                size=35,
                color='#181818',
                ax=ax4)

        ########################################################################################################################################################

        ax5 = plt.subplot(
                        gspec[0, 5],
                )

        cols = role

        #PLAYER 1
        pl1 = df.loc[df['Player'] == data.get('Players')[3]]
        val1 = pl1[cols].values[0]

        # PLAYER 2
        pl2 = df.loc[df['Player'] == playerName]
        val2 = pl2[cols].values[0]

        values = [val1, val2]
        #Obtenção do alcance minimo e máximo dos valores
        ranges = [(df[col].min(), df[col].max()) for col in cols]

        #Criação do radar #181818
        radar = Radar(background_color="#181818", patch_color='#181818', range_color="#181818", label_color="#181818", label_fontsize=0, range_fontsize=0)
        fig, ax5 = radar.plot_radar(ranges=ranges, 
                                params=cols, 
                                values=values, 
                                radar_color=['#2d92df', '#548135'],
                                figax=(fig, ax5),
                                end_size=0, end_color="#181818",
                                compare=True)

        age = pl1['Age'].unique()
        age = age.tolist()
        age = age[0]

        tier = pl1['Tier'].unique()
        tier = tier.tolist()
        tier = tier[0]

        ax_text(x = -8, y = -27,
                s= data.get('Players')[3],
                size=48,
                color='#181818',
                ax=ax5)

        ax_text(x = -4, y = -34,
                s='(' + str(age) + ')',
                size=40,
                color='#181818',
                ax=ax5)

        team = pl1['Team'].unique()
        team = team.tolist()
        team = team[0]

        ax_text(x = -5, y = -19,
                s=team + ' ' + '(' + str(tier) + ')',
                size=35,
                color='#181818',
                ax=ax5)

        ########################################################################################################################################################

        ax7 = plt.subplot(
                        gspec[0, 6],
                )

        cols = role

        #PLAYER 1
        pl1 = df.loc[df['Player'] == data.get('Players')[4]]
        val1 = pl1[cols].values[0]

        # PLAYER 2
        pl2 = df.loc[df['Player'] == playerName]
        val2 = pl2[cols].values[0]

        values = [val1, val2]
        #Obtenção do alcance minimo e máximo dos valores
        ranges = [(df[col].min(), df[col].max()) for col in cols]

        #Criação do radar chart
        radar = Radar(background_color="#181818", patch_color='#181818', range_color="#181818", label_color="#181818", label_fontsize=0, range_fontsize=0)
        fig, ax7 = radar.plot_radar(ranges=ranges, 
                                params=cols, 
                                values=values, 
                                radar_color=['#2d92df', '#548135'],
                                figax=(fig, ax7),
                                end_size=0, end_color="#181818",
                                compare=True)

        age = pl1['Age'].unique()
        age = age.tolist()
        age = age[0]

        tier = pl1['Tier'].unique()
        tier = tier.tolist()
        tier = tier[0]

        ax_text(x = -8, y = -27,
                s= data.get('Players')[4],
                size=48,
                color='#181818',
                ax=ax7)

        ax_text(x = -4, y = -34,
                s='(' + str(age) + ')',
                size=40,
                color='#181818',
                ax=ax7)

        team = pl1['Team'].unique()
        team = team.tolist()
        team = team[0]

        ax_text(x = -5, y = -19,
                s=team + ' ' + '(' + str(tier) + ')',
                size=35,
                color='#181818',
                ax=ax7)

########################################################################################################################################################

        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
        [{"color": "#548135","fontweight": 'bold'}]

        fig_text(x = 0.15, y = 0.8,
                s= 'Similar to:' + ' ' + f'<{playerName}>',
                size=100,
                highlight_textprops = highlight_textprops,
                color='#181818')

########################################################################################################################################################

        subtitle = df.loc[df['Player'] == playerName]

        team = subtitle.Team.unique()

        team = team.tolist()

        team = team[0]

########################################################################################################################################################

        league = subtitle.Comp.unique()

        league = league.tolist()

        league = league[0]

########################################################################################################################################################

        season = subtitle.Season.unique()

        season = season.tolist()

        season = season[0]

########################################################################################################################################################

        fig_text(x = 0.15, y = 0.768,
                s= team + ' |' + league + ' |' + season,
                size=35,
                color='#181818',
                ax=ax6)

        ax_text(x = 0.15, y = -0.70,
                s= 'Visualization made by: menesesp20',
                size=30,
                color='#181818',
                ax=ax6)

        ax_text(x = 0.15, y = -1,
                s= 'Data from WyScout',
                size=25,
                color='#181818',
                ax=ax6)

################################################################################################################################################

def createTier(league):
    if league in tier0:
        return 0

    elif league in tier1:
        return 1

    elif league in tier2:
        return 2

    elif league in tier3:
        return 3

    elif league in tier4:
        return 4

"""
df['Tier'] = df.apply(lambda x: createTier(x.Comp), axis=1)

df['Status'] = pd.qcut(df['Minutes played'], np.linspace(0,1,11), labels=np.linspace(0.1,1,10))

df['Status'] = df['Status'].astype(float)

def gameTime(col):
    if col >= 0.8:
        return 'Titular'

    elif (col >= 0.5) & (col < 0.8):
        return 'Suplente'

    elif col < 0.5:
        return 'Reserva'

df['Status'] = df.apply(lambda x: gameTime(x['Status']), axis=1)
"""

################################################################################################################################################

def tierLeague(df, playerName):

    df = df.loc[df.Player == playerName].reset_index(drop=True)

    tier = df.Tier.unique()
    tier = tier.tolist()
    tier = tier[0]

    rating = df[['Box Forward', 'False 9', 'Target Man', 'Advanced Forward', 'Ball Winner', 'Deep Lying Playmaker',
                                                        'Attacking Playmaker', 'Box-to-box','Attacking FB', 'Defensive FB', 'Full Back CB',
                                                        'Wing Back', 'Inverted Wing Back', 'Stopper', 'Aerial CB', 'Ball Playing CB',
                                                        'Ball Carrying CB', 'Banda', 'Defensivo', 'Falso', 'Por dentro']].max(axis=0).sort_values(ascending=False)

    rating = rating[0]

    if tier == 0:
        rating = round(rating * 1, 2)

        return rating

    if tier == 1:
        rating = round(rating * 0.95, 2)

        return rating

    elif tier == 2:
        rating = round(rating * 0.88, 2)

        return rating

    elif tier == 3:
        rating = round(rating * 0.8, 2)

        return rating

    elif tier == 4:
        rating = round(rating * 0.75, 2)

        return rating

################################################################################################################################################

def gameTime(col):
    if col >= 0.8:
        return 'Titular'

    elif (col >= 0.5) & (col < 0.8):
        return 'Suplente'

    elif col < 0.5:
        return 'Reserva'

################################################################################################################################################

def positionMain(col):
    if (col.__contains__('RB')) | (col.__contains__('LB')) | (col.__contains__('RWB')) | (col.__contains__('LWB')):
        return 'Full Back'

    elif col.__contains__('CB') | col.__contains__('RCB') | col.__contains__('LCB'):
        return 'Center Back'

    elif (col.__contains__('CF')):
        return 'Forward'

    elif (col.__contains__('RCMF')) | (col.__contains__('LCMF')) | (col.__contains__('LDMF')) | (col.__contains__('RDMF')) | (col.__contains__('AMF')) | (col.__contains__('DMF')):
        return 'Midfield'

    elif (col.__contains__('LW')) | (col.__contains__('LM')) | (col.__contains__('RM')) | (col.__contains__('RW')) | (col.__contains__('RAMF')) | (col.__contains__('LAMF')) | (col.__contains__('RWF')) | (col.__contains__('LWF')):
        return 'Winger'

################################################################################################################################################

def evaluation(df, league, club, playerName, role):

    # plot
    fig, ax = plt.subplots(figsize=(10, 8))

    #Set color background outside the graph
    fig.set_facecolor('#E8E8E8')

    #Set color background inside the graph
    ax.set_facecolor('#E8E8E8')

    ax.tick_params(axis='x', colors='#181818', labelsize=12)
    ax.tick_params(axis='y', colors='#181818', labelsize=12)

    ax.spines['bottom'].set_color('#181818')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#181818')
    ax.spines['right'].set_visible(False)

    test = df.loc[df.Player == playerName].sort_values('Season', ascending=True).reset_index(drop=True)
    test.drop_duplicates(subset=['Season'], keep='last', inplace=True)

    y = test[role]
    x = test['Season']

    ax.set_ylim([0, 100])
    
    ax.plot(x, y,  marker='.', markersize=12)

    plt.title(playerName + ' Evaluation', c='#181818', fontsize=35, y=1.09)

    plt.ylabel('menesesp20 VALUE', color='#181818', size=11)
    plt.xlabel('SEASON', color='#181818', size=11)

    fig = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + league + '/' + club + '/' + playerName + '.png', fig=fig, left=0.12, bottom=0.93, width=0.12, height=0.11)

    plt.show()

################################################################################################################################################

def beeswarm_Individual(df, event, ax, Player, Player2=None, Player3=None, Player4=None, Player5=None, Player6=None, Player7=None, Player8=None):

  #set default colors
  text_color = '#181818'
  background = '#E8E8E8'

  player = df.loc[(df['Player'] == Player)]
  league = player.Comp.unique()
  league = league.tolist()
  league = league[0]

  player2 = df.loc[(df['Player'] == Player2)]

  player3 = df.loc[(df['Player'] == Player3)]

  player4 = df.loc[(df['Player'] == Player4)]

  player5 = df.loc[(df['Player'] == Player5)]

  player6 = df.loc[(df['Player'] == Player6)]

  player7 = df.loc[(df['Player'] == Player7)]

  player8 = df.loc[(df['Player'] == Player8)]

  #set up our base layer
  mpl.rcParams['xtick.color'] = '#E8E8E8'
  mpl.rcParams['ytick.color'] = '#E8E8E8'

  ax.grid(ls='dotted', lw=.5, color='#E8E8E8', axis='y', zorder=2)
  spines = ['top', 'bottom', 'left', 'right']
  for x in spines:
      if x in spines:
          ax.spines[x].set_visible(False)


  #BeeSwarm

  df = df.loc[(df['Comp'] == league) | (df['Player'] == Player2) | (df['Player'] == Player3) |
              (df['Player'] == Player4) | (df['Player'] == Player5) | (df['Player'] == Player6) | (df['Player'] == Player7)]

  besswarm = sns.swarmplot(x=event, data=df, color='#181818', alpha=0.8, zorder=2)

  besswarm.set(xlabel=None)

  #plot Player
  for i in range(len(player)):
    plt.scatter(x=player[event].values[i], y=0, c='#c70216', edgecolor='black', s=150, zorder=2, label=f'{Player}')
    
  #plot player2
  for i in range(len(player2)):
    plt.scatter(x=player2[event].values[i], y=0, c='#00659c', edgecolor='black', s=150, zorder=2, label=f'{Player2}')

  #plot player3
  for i in range(len(player3)):
    plt.scatter(x=player3[event].values[i], y=0, c='#03b976', edgecolor='black',s=150, zorder=2, label=f'{Player3}')

  #plot player4
  for i in range(len(player4)):
    plt.scatter(x=player4[event].values[i], y=0, c='pink', edgecolor='black', s=150, zorder=2, label=f'{Player4}')

  #plot player5
  for i in range(len(player5)):
    plt.scatter(x=player5[event].values[i], y=0, c='yellow', edgecolor='black', s=150, zorder=2, label=f'{Player5}')

  #plot player6
  for i in range(len(player6)):
    plt.scatter(x=player6[event].values[i], y=0, c='#fb8c04', edgecolor='black', s=150, zorder=2, label=f'{Player6}')

  #plot player7
  for i in range(len(player7)):
    plt.scatter(x=player7[event].values[i], y=0, c='#00bbf9', edgecolor='black', s=150, zorder=2, label=f'{Player7}')

  #plot player7
  for i in range(len(player8)):
    plt.scatter(x=player8[event].values[i], y=0, c='#cad2c5', edgecolor='black', s=150, zorder=2, label=f'{Player8}')

  plt.title(event, c=text_color, fontsize=20)
  #Criação da legenda
  l = plt.legend(facecolor='#181818', framealpha=.05, labelspacing=.7, prop={'size': 8})
  #Ciclo FOR para atribuir a white color na legend
  for text in l.get_texts():
      text.set_color("#181818")

################################################################################################################################################

def beeSwarm(wyscout):
        fig = plt.figure(figsize=(20, 15), dpi=500, facecolor = '#E8E8E8')
        gspec = gridspec.GridSpec(
        ncols=2, nrows=3, wspace = 0.5
        )

        background = '#E8E8E8'

        # LEFT
        ax1 = plt.subplot(
                        gspec[0, 0],
                )
        ax1.patch.set_facecolor(background)
        ax1.axis('off')
        beeswarm_Individual(wyscout, Forward[0], ax1, 'A. da Silva', 'K. Kondo', 'A. Dowds', 'P. Knudsen', 'L. Vardanyan')


        ax2 = plt.subplot(
                        gspec[1, 0],
                )
        ax2.patch.set_facecolor(background)
        ax2.axis('off')
        beeswarm_Individual(wyscout, Forward[1], ax2, 'A. da Silva', 'K. Kondo', 'A. Dowds', 'P. Knudsen', 'L. Vardanyan')


        ax3 = plt.subplot(
                        gspec[2, 0],
                        
                )
        ax3.patch.set_facecolor(background)
        ax3.axis('off')
        beeswarm_Individual(wyscout, Forward[2], ax3, 'A. da Silva', 'K. Kondo', 'A. Dowds', 'P. Knudsen', 'L. Vardanyan')


        # RIGHT
        ax4 = plt.subplot(
                        gspec[0, 1],
                )
        ax4.patch.set_facecolor(background)
        ax4.axis('off')
        beeswarm_Individual(wyscout, Forward[3], ax4, 'A. da Silva', 'K. Kondo', 'A. Dowds', 'P. Knudsen', 'L. Vardanyan')


        ax5 = plt.subplot(
                        gspec[1, 1],
                )
        ax5.patch.set_facecolor(background)
        ax5.axis('off')
        beeswarm_Individual(wyscout, Forward[4], ax5, 'A. da Silva', 'K. Kondo', 'A. Dowds', 'P. Knudsen', 'L. Vardanyan')


        ax6 = plt.subplot(
                        gspec[2, 1],
                )
        ax6.patch.set_facecolor(background)
        ax6.axis('off')
        beeswarm_Individual(wyscout, Forward[6], ax6, 'A. da Silva', 'K. Kondo', 'A. Dowds', 'P. Knudsen', 'L. Vardanyan')

################################################################################################################################################

def PizzaChart(df, cols, playerName, league):
    # parameter list
    params = cols

    playerDF = df.loc[(df.Player == playerName) & (df.Comp == league)]

    league = playerDF.Comp.unique()

    league = league.tolist()

    league = league[0]

    position = playerDF['Position'].unique()

    position = position.tolist()

    position = position[0]
    if ', ' in position:
        position = position.split(', ')[0]

    marketValue = playerDF['Market value'].unique()

    marketValue = marketValue.tolist()
    
    marketValue = marketValue[0]

    df = df.loc[(df['Comp'] == league) & (df['Position'].str.contains(position))].reset_index(drop=True)

    player = df.loc[(df['Player'] == playerName) & (df['Comp'] == league)][cols].reset_index()
    player = list(player.loc[0])
    player = player[1:]

    values = []
    for x in range(len(params)):   
        values.append(math.floor(stats.percentileofscore(df[params[x]], player[x])))

    for n,i in enumerate(values):
        if i == 100:
            values[n] = 99

    if cols == Forward:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 4 + ["#fb8c04"] * 4 + ["#eb04e3"] * 4
        text_colors = ["#F2F2F2"] * 12

    elif cols == Winger:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 3 + ["#fb8c04"] * 8 + ["#eb04e3"] * 2
        text_colors = ["#F2F2F2"] * 13

    elif cols == defensive_Midfield:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 4 + ["#fb8c04"] * 4 + ["#eb04e3"] * 5
        text_colors = ["#F2F2F2"] * 13
        
    elif cols == Midfield:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 4 + ["#fb8c04"] * 8 + ["#eb04e3"] * 3
        text_colors = ["#F2F2F2"] * 15

    elif cols == full_Back:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 6 + ["#fb8c04"] * 5 + ["#eb04e3"] * 4
        text_colors = ["#F2F2F2"] * 15

    elif cols == center_Back:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 3 + ["#fb8c04"] * 4 + ["#eb04e3"] * 7
        text_colors = ["#F2F2F2"] * 14

    elif cols == offensive_Midfield:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 4 + ["#fb8c04"] * 8 + ["#eb04e3"] * 4
        text_colors = ["#F2F2F2"] * 16

    # instantiate PyPizza class
    baker = PyPizza(
        params=params,                  # list of parameters
        background_color="#1b1b1b",     # background color
        straight_line_color="#000000",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_color="#000000",    # color for last line
        last_circle_lw=1,               # linewidth of last circle
        other_circle_lw=0,              # linewidth for other circles
        inner_circle_size=20            # size of inner circle
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        values,                          # list of values
        figsize=(15, 10),                # adjust the figsize according to your need
        color_blank_space="same",        # use the same color to fill blank space
        slice_colors=slice_colors,       # color for individual slices
        value_colors=text_colors,        # color for the value-text
        value_bck_colors=slice_colors,   # color for the blank spaces
        blank_alpha=0.4,                 # alpha for blank-space colors
        kwargs_slices=dict(
            edgecolor="#000000", zorder=2, linewidth=1
        ),                               # values to be used when plotting slices
        kwargs_params=dict(
            color="#F2F2F2", fontsize=10,
            va="center"
        ),                               # values to be used when adding parameter labels
        kwargs_values=dict(
            color="#F2F2F2", fontsize=11,
            zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
            )
        )                                # values to be used when adding parameter-values labels
    )

    if cols == Forward:

        fig_text(s =  'Forward Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    elif cols == Winger:

        fig_text(s =  'Winger Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    elif cols == defensive_Midfield:

        fig_text(s =  'Defensive Midfield Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    elif cols == Midfield:

        fig_text(s =  'Midfield Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    elif cols == full_Back:

        fig_text(s =  'Full Back Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)
    elif cols == center_Back:

        fig_text(s =  'Center Back Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    elif cols == offensive_Midfield:

        fig_text(s =  'Offensive Midfield Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    ###########################################################################################################

    fig_text(s =  playerName,
             x = 0.5, y = 1.12,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=50);

    if playerName != 'David Neres':

        fig_text(s =  'Percentile Rank | ' + league + ' | Pizza Chart | Season 2021-22',
                x = 0.5, y = 1.03,
                color='#F2F2F2',
                fontweight='bold', ha='center',
                fontsize=14);

    elif playerName == 'David Neres':

        fig_text(s =  'Percentile Rank | ' + league + ' | Pizza Chart | Calendar Year 2021',
                x = 0.5, y = 1.03,
                color='#F2F2F2',
                fontweight='bold', ha='center',
                fontsize=14);

    #fig_text(s =  str(marketValue),
    #         x = 0.5, y = 1.02,
    #         color='#F2F2F2',
    #         fontweight='bold', ha='center',
    #         fontsize=18);

    # add credits
    CREDIT_1 = "data: WyScout"
    CREDIT_2 = "made by: @menesesp20"
    CREDIT_3 = "inspired by: @Worville, @FootballSlices, @somazerofc & @Soumyaj15209314"


    # CREDITS
    fig_text(s =  f"{CREDIT_1}\n{CREDIT_2}\n{CREDIT_3}",
             x = 0.35, y = 0.02,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8);

    # Attacking
    fig_text(s =  'Attacking',
             x = 0.41, y = 0.988,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=16);

    # Possession
    fig_text(s =  'Possession',
             x = 0.535, y = 0.988,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=16);

    # Defending
    fig_text(s =  'Defending',
             x = 0.665, y = 0.988,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=16);

    # add rectangles
    fig.patches.extend([
        plt.Rectangle(
            (0.34, 0.97), 0.025, 0.021, fill=True, color="#2d92df",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.47, 0.97), 0.025, 0.021, fill=True, color="#fb8c04",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.60, 0.97), 0.025, 0.021, fill=True, color="#eb04e3",
            transform=fig.transFigure, figure=fig
        ),
    ])

    # add image
    add_image('C:/Users/menes/Documents/Data Hub/Images/SWL LOGO.png', fig, left=0.462, bottom=0.436, width=0.10, height=0.132)

################################################################################################################################################

def role_Chart(df, playerName, pos, league):
    # parameter and value list
    if pos == 'Goalkepper':
        df = df.loc[df.Position == 'GK'].reset_index(drop=True)
        params = ['1v1 GK', 'BOX GK', 'Ball Playing GK']

    elif pos == 'Center Back':
        params = ['Ball Playing CB', 'Ball Carrying CB', 'Stopper', 'Aerial CB']

        df = df.loc[(df.Position.str.contains('CB')) & (df.Comp == league)].reset_index(drop=True)

        playerAbility(df)

        RoleCenterBack(df)

        szcore_df(df)

    elif pos == 'Full Back':
        params = ['Attacking FB', 'Wing Back', 'Inverted Wing Back', 'Defensive FB', 'Full Back CB']

        df = df.loc[((df.Position.str.contains('RB')) | (df.Position.str.contains('RWB')) | (df.Position.str.contains('LWB')) | (df.Position.str.contains('LB')))].reset_index(drop=True)

        playerAbility(df)

        roleFullBack(df)

        szcore_df(df)

    elif pos == 'Midfield':
        params = ['Box-to-box', 'Attacking Playmaker', 'Deep Lying Playmaker', 'Ball Winner', 'Media Punta llegador']

        df = df.loc[((df.Position.str.contains('RCMF')) | (df.Position.str.contains('LCMF')) | (df.Position.str.contains('LDMF')) | (df.Position.str.contains('RDMF')) |
                            (df.Position.str.contains('AMF')) | (df.Position.str.contains('DMF')) | (df.Position.str.contains('RAMF')))].reset_index(drop=True)

        playerAbility(df)

        RoleMidfield(df)

        szcore_df(df)

    elif pos == 'Winger':
        params = ['Banda' , 'Por dentro', 'Defensivo', 'Falso']

        df = df.loc[(df.Position.str.contains('LW')) | (df.Position.str.contains('RW')) | (df.Position.str.contains('LAMF')) |
                     (df.Position.str.contains('RAMF')) | (df.Position.str.contains('LWF')) | (df.Position.str.contains('RWF'))].reset_index(drop=True)

        playerAbility(df)

        RoleExtremo(df)

        szcore_df(df)

    elif pos == 'Forward':
        params = ['Advanced Forward', 'Target Man', 'False 9', 'Box Forward']
        df = df.loc[(df.Position.str.contains('CF'))].reset_index(drop=True)

        playerAbility(df)

        RoleForward(df)

        szcore_df(df)
    
    players = df.loc[df['Player'] == playerName]

    club = players.Team.unique()
    club = club.tolist()
    club = club[0]

    league = players.Comp.unique()
    league = league.tolist()
    league = league[0]

    #Valores que pretendemos visualizar no radar chart, acedemos ao index 0 para obtermos os valores dentro da lista correta

    df = df.loc[(df['Comp'] == league)].reset_index(drop=True)

    player = df.loc[(df['Player'] == playerName) & (df['Comp'] == league)][params].reset_index()
    player = list(player.loc[0])
    player = player[1:]
    
    values = []
    for x in range(len(params)):   
        values.append(math.floor(stats.percentileofscore(df[params[x]], player[x])))

    for n,i in enumerate(values):
        if i == 100:
            values[n] = 99

    # instantiate PyPizza class
    baker = PyPizza(
        params=params,                  # list of parameters
        straight_line_color="#F2F2F2",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        straight_line_limit=100.0,        # max limit of straight lines
        last_circle_lw=0,               # linewidth of last circle
        other_circle_lw=0,              # linewidth for other circles
        inner_circle_size=0.4,          # size of inner circle
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        values,                     # list of values
        figsize=(8, 8),             # adjust figsize according to your need
        color_blank_space="same",   # use same color to fill blank space
        blank_alpha=0.4,            # alpha for blank-space colors
        param_location=104.7,         # where the parameters will be added
        kwargs_slices=dict(
            facecolor="#043484", edgecolor="#F2F2F2",
            zorder=2, linewidth=1
        ),                          # values to be used when plotting slices
        kwargs_params=dict(
            color="#000000", fontsize=12,
            va="center"
        ),                          # values to be used when adding parameter
        kwargs_values=dict(
            color="#E8E8E8", fontsize=12,
            zorder=3,
            bbox=dict(
                edgecolor="#E8E8E8", facecolor="#043484",
                boxstyle="round,pad=0.2", lw=1
            )
        )                           # values to be used when adding parameter-values
    )

    fig.set_facecolor('#E8E8E8')

    fig_text(s =  playerName,
                x = 0.5, y = 1.07,
                color='#181818',
                fontweight='bold', ha='center',
                fontsize=28)

    fig_text(s =  'Rank vs Players Position | Season 21/22',
                x = 0.5, y = 1.02,
                color='#181818',
                fontweight='bold', ha='center',
                fontsize=12)

################################################################################################################################################

def radar_chartPhysical(physical, player, cols):

  color = ['#890e3e', '#f9f1e7']

  from soccerplots.radar_chart import Radar

  #Atribuição do jogador a colocar no gráfico
  players = physical.loc[(physical['player'] == player)].reset_index(drop=True)

    #####################################################################################################################
    #####################################################################################################################

  #Valores que pretendemos visualizar no radar chart, acedemos ao index 0 para obtermos os valores dentro da lista correta
  values = players[cols].values[0]
  #Obtenção do alcance minimo e máximo dos valores

  ranges = [(physical[col].min(), physical[col].max()) for col in cols]
  
  #Atribuição dos valores aos titulos e respetivos tamanhos e cores
  title = dict(
  title_name = player,
  title_color = color[0],
  title_fontsize = 40,
  subtitle_fontsize = 30,
  subtitle_name='Physical data',
  subtitle_color='#181818',
  )

    #team_player = df[col_name_team].to_list()

    #dict_team ={'Dortmund':['#ffe011', '#000000'],
              #'Nice':['#cc0000', '#000000'],}

    #color = dict_team.get(team_player[0])

    ## endnote 
  endnote = "Visualization made by: Pedro Meneses(@menesesp20)"

  #Criação do radar chart
  fig, ax = plt.subplots(figsize=(18,15))
  radar = Radar(background_color="#E8E8E8", patch_color="#181818", range_color="#181818", label_color="#181818", label_fontsize=20, range_fontsize=12)
  fig, ax = radar.plot_radar(ranges=ranges, 
                               params=cols, 
                               values=values, 
                               radar_color=color,
                               figax=(fig, ax),
                               image_coord=[0.464, 0.81, 0.1, 0.075],
                               title=title,
                               endnote=endnote)

  # add image
  add_image('Images/WorldCup_Qatar.png', fig, left=0.68, bottom=0.83, width=0.1, height=0.12)

  fig.set_facecolor('#E8E8E8')

  plt.savefig('assets/radarPhysical' + player + '.png', dpi=300)

  return app.get_asset_url('radarPhysical' + player + '.png')

################################################################################################################################################

def PizzaChartPhysical(physical, playerName, cols):
    # parameter list
    params = cols

    player = physical.loc[(physical['player'] == playerName)][cols].reset_index()
    player = list(player.loc[0])
    player = player[1:]

    values = []
    for x in range(len(params)):   
        values.append(math.floor(stats.percentileofscore(physical[params[x]], player[x])))

    for n,i in enumerate(values):
        if i == 100:
            values[n] = 99

    # color for the slices and text
    slice_colors = ["#2d92df"] * 2 + ["#fb8c04"] * 6 + ["#eb04e3"] * 2
    text_colors = ["#F2F2F2"] * 10

    # instantiate PyPizza class
    baker = PyPizza(
        params=params,                  # list of parameters
        background_color="#E8E8E8",     # background color
        straight_line_color="#000000",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_color="#000000",    # color for last line
        last_circle_lw=1,               # linewidth of last circle
        other_circle_lw=0,              # linewidth for other circles
        inner_circle_size=20            # size of inner circle
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        values,                          # list of values
        figsize=(15, 10),                # adjust the figsize according to your need
        color_blank_space="same",        # use the same color to fill blank space
        slice_colors=slice_colors,       # color for individual slices
        value_colors=text_colors,        # color for the value-text
        value_bck_colors=slice_colors,   # color for the blank spaces
        blank_alpha=0.4,                 # alpha for blank-space colors
        kwargs_slices=dict(
            edgecolor="#000000", zorder=2, linewidth=1
        ),                               # values to be used when plotting slices
        kwargs_params=dict(
            color="#181818", fontsize=10,
            va="center"
        ),                               # values to be used when adding parameter labels
        kwargs_values=dict(
            color="#181818", fontsize=11,
            zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
            )
        )                                # values to be used when adding parameter-values labels
    )

    fig_text(s =  'Physical Template',
            x = 0.253, y = 0.035,
            color='#181818',
            fontweight='bold', ha='center',
            fontsize=5)

    ###########################################################################################################

    fig_text(s =  playerName,
             x = 0.5, y = 1.12,
             color='#181818',
             fontweight='bold', ha='center',
             fontsize=14);

    # add credits
    CREDIT_1 = "data: FIFA"
    CREDIT_2 = "made by: @menesesp20"
    CREDIT_3 = "inspired by: @Worville, @FootballSlices, @somazerofc & @Soumyaj15209314"


    # CREDITS
    fig_text(s =  f"{CREDIT_1}\n{CREDIT_2}\n{CREDIT_3}",
             x = 0.35, y = 0.02,
             color='#181818',
             fontweight='bold', ha='center',
             fontsize=5);

    # Attacking
    fig_text(s =  'Speed',
             x = 0.41, y = 0.988,
             color='#181818',
             fontweight='bold', ha='center',
             fontsize=8);

    # Possession
    fig_text(s =  'Distance',
             x = 0.535, y = 0.988,
             color='#181818',
             fontweight='bold', ha='center',
             fontsize=8);

    # Defending
    fig_text(s =  'Sprint',
             x = 0.665, y = 0.988,
             color='#181818',
             fontweight='bold', ha='center',
             fontsize=8);

    # add rectangles
    fig.patches.extend([
        plt.Rectangle(
            (0.34, 0.97), 0.025, 0.021, fill=True, color="#2d92df",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.47, 0.97), 0.025, 0.021, fill=True, color="#fb8c04",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.60, 0.97), 0.025, 0.021, fill=True, color="#eb04e3",
            transform=fig.transFigure, figure=fig
        ),
    ])

    # add image
    add_image('Images/WorldCup_Qatar.png', fig, left=0.475, bottom=0.452, width=0.075, height=0.105)

    plt.savefig('assets/pizzaPhysical' + playerName + '.png', dpi=300)

    return app.get_asset_url('pizzaPhysical' + playerName + '.png')

################################################################################################################################################

def pizza_ComparePhysical(physical, playerName, playerName2, cols):
    
    params = cols
    
    color = ['#890e3e', '#f9f1e7']
    
    player = physical.loc[physical['player'] == playerName][cols].reset_index()
    player = list(player.loc[0])
    player = player[1:]

    values = []
    for x in range(len(params)):   
        values.append(math.floor(stats.percentileofscore(physical[params[x]], player[x])))

    for n,i in enumerate(values):
        if i == 100:
            values[n] = 99

    ###########################################################################################

    player2 = physical.loc[physical['player'] == playerName2][cols].reset_index()
    player2 = list(player2.loc[0])
    player2 = player2[1:]

    values_2 = []
    for x in range(len(params)):   
        values_2.append(math.floor(stats.percentileofscore(physical[params[x]], player2[x])))

    for n,i in enumerate(values_2):
        if i == 100:
            values_2[n] = 99

    # instantiate PyPizza class
    baker = PyPizza(
        params=params,                  # list of parameters
        background_color="#E8E8E8",     # background color
        straight_line_color="#000000",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_lw=1,               # linewidth of last circle
        last_circle_color="#000000",    # color of last circle
        other_circle_ls="-.",           # linestyle for other circles
        other_circle_lw=1,              # linewidth for other circles
        inner_circle_size=20            
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        values,                     # list of values
        compare_values=values_2,    # comparison values
        figsize=(15, 10),             # adjust figsize according to your need
        kwargs_slices=dict(
            facecolor=color[0], edgecolor="#000000",
            zorder=2, linewidth=1
        ),                          # values to be used when plotting slices
        kwargs_compare=dict(
            facecolor=color[1], edgecolor="#000000",
            zorder=2, linewidth=1,
        ),
        kwargs_params=dict(
            color="#181818",
            fontsize=12,
            va="center"
        ),                          # values to be used when adding parameter
        kwargs_values=dict(
            color="#181818",
            fontsize=12,
            zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor=color[0],
                boxstyle="round,pad=0.2", lw=1
            )
        ),                          # values to be used when adding parameter-values labels
        kwargs_compare_values=dict(
            color="#181818",
            fontsize=12,
            zorder=3,
            bbox=dict(edgecolor="#000000", facecolor=color[1], boxstyle="round,pad=0.2", lw=1)
        ),                          # values to be used when adding parameter-values labels
    )


    fig_text(s = 'Physical Template | World Cup Catar 2022',
            x = 0.515, y = 1.12,
            color='#181818',
            fontweight='bold', ha='center',
            fontsize=14)

    ###########################################################################################################

    # add credits
    CREDIT_1 = "data: WyScout"
    CREDIT_2 = "made by: @menesesp20"
    CREDIT_3 = "inspired by: @Worville, @FootballSlices, @somazerofc & @Soumyaj15209314"


    # CREDITS
    fig_text(s =  f"{CREDIT_1}\n{CREDIT_2}\n{CREDIT_3}",
             x = 0.35, y = 0.02,
             color='#181818',
             fontweight='bold', ha='center',
             fontsize=5);

    # Attacking
    fig_text(s =  playerName,
             x = 0.47, y = 0.988,
             color='#181818',
             fontweight='bold', ha='center',
             fontsize=12);

    # Possession
    fig_text(s =  playerName2,
             x = 0.605, y = 0.988,
             color='#181818',
             fontweight='bold', ha='center',
             fontsize=12);

    # add rectangles
    fig.patches.extend([
        plt.Rectangle(
            (0.40, 0.97), 0.025, 0.021, fill=True, color=color[0],
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.52, 0.97), 0.025, 0.021, fill=True, color=color[1],
            transform=fig.transFigure, figure=fig
        )
        ])

    # add image
    add_image('Images/WorldCup_Qatar.png', fig, left=0.475, bottom=0.452, width=0.075, height=0.105)

    plt.savefig('assets/pizzaComparePhysical' + playerName + '.png', dpi=300)

    return app.get_asset_url('pizzaComparePhysical' + playerName + '.png')

################################################################################################################################################
#--------------------------------------------------- DEFENSIVE --------------------------------------------------------------------------------
################################################################################################################################################

def RoleCenterBackDefensive(df):

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10


    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Stopper'] = round(((df['Defensive Ability'] * 27.5) + (df['positioning Defence'] * 37.5) + (df[ 'Aerial'] * 20) +
                           (df[ 'ballPlaying'] * 5) + (df['attackingThreat Defence'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Aerial CB'] =  round(((df['Defensive Ability'] * 20) + (df['positioning Defence'] * 25) +
                             (df[ 'Aerial'] * 45) + (df['attackingThreat Defence'] * 10)) / 10, 2)
 
    ##########################################################################################################################################################################

    df['Ball Playing CB'] =  round(((df['Defensive Ability'] * 27.5) + (df['positioning Defence'] * 17.5) + (df[ 'Aerial'] * 10) +
                                    (df[ 'ballPlaying'] * 30) + (df[ 'progressiveRuns'] * 10) + (df['attackingThreat Defence'] * 5)) / 10, 2)

    ##########################################################################################################################################################################

    df['Ball Carrying CB'] =  round(((df['Defensive Ability'] * 25) + (df[ 'positioning Defence'] * 20) + (df[ 'Aerial'] * 15) +
                            (df[ 'ballPlaying'] * 10) + (df[ 'progressiveRuns'] * 25) + (df[ 'attackingThreat Defence'] * 5)) / 10, 2)

################################################################################################################################################

def roleFullBackDefensive(df):

    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Attacking FB'] =  round(((df['crossing'] * 17.5) + (df['dribbles'] * 10) + (df['positioning Defence'] * 20) +
                                (df['progressiveRuns'] * 10) + (df['defending1v1'] * 20) +
                                (df['decisionMake'] * 12.5) + (df[ 'touchQuality'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Wing Back'] =  round(((df['crossing'] * 25) + (df[ 'defending1v1'] * 20) + (df[ 'positioning Defence'] * 25) +
                              (df[ 'progressiveRuns'] * 15) + (df[ 'decisionMake'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Inverted Wing Back'] =  round(((df['dribbles'] * 10) + (df['defending1v1'] * 30) + (df['positioning Midfield'] * 20) +
                                      (df['positioning Defence'] * 25) + (df['decisionMake'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Defensive FB'] =  round(((df['crossing'] * 10) + (df[ 'defending1v1'] * 40) +
                                (df['positioning Defence'] * 30) + (df[ 'Aerial'] * 20)) / 10, 2)

    ##########################################################################################################################################################################

    df['Full Back CB'] =  round(((df['crossing'] * 10) + (df[ 'defending1v1'] * 45) +
                                (df['positioning Defence'] * 35) + (df[ 'Aerial'] * 10)) / 10, 2)

################################################################################################################################################

def RoleMidfieldDefensive(df):

    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Ball Winner'] =  round(((df['positioning Midfield'] * 35) + (df['defending1v1'] * 30) +
                               (df['ballPlaying'] * 5) + (df['Aerial'] * 30)) / 10, 2)

    ##########################################################################################################################################################################

    df['Deep Lying Playmaker'] =  round(((df['positioning Midfield'] * 35) + (df['decisionMake'] * 15) +
                                        (df['Aerial'] * 25) + (df['defending1v1'] * 15) + (df['ballPlaying Deep'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Attacking Playmaker'] =  round(((df['positioning Midfield'] * 30) + (df['progressiveRuns'] * 15) +
                                    (df['decisionMake'] * 15) + (df['touchQuality'] * 10) + (df['dribbles'] * 15) + (df['ballPlaying'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Media Punta llegador'] =  round(((df['attackingThreat'] * 10) + (df['progressiveRuns'] * 10) + (df['positioning Midfield'] * 20) +
                                    (df['decisionMake'] * 20) + (df['touchQuality'] * 15) + (df['dribbles'] * 10) + (df['ballPlaying'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Box-to-box'] =  round(((df['Pass_Forward'] * 25) + (df['defending1v1'] * 40) + (df['runs'] * 30))/ 10, 2)

################################################################################################################################################

def RoleExtremoDefensivo(df):
    
    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Banda'] =  round(((df['progressiveRuns'] * 25) + (df['decisionMake'] * 10) +
                          (df['defending1v1'] * 25) + (df['dribbles'] * 7.5) + (df['crossing'] * 22.5) +
                          (df['attackingThreat'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Por dentro'] =  round(((df['progressiveRuns'] * 20) + (df['defending1v1'] * 17.5) + (df['decisionMake'] * 12.5) +
                                (df['touchQuality'] * 15) + (df['dribbles'] * 20) +
                                (df['attackingThreat'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Defensivo'] =  round(((df['crossing'] * 20) + (df['dribbles'] * 10) + (df['positioning Midfield'] * 25) +
                              (df['decisionMake'] * 15) + (df['defending1v1'] * 30)) / 10, 2)

    ##########################################################################################################################################################################

    df['Falso'] =  round(((df['attackingThreat'] * 28.5) + (df['dribbles'] * 15) + (df['defending1v1'] * 20) +
                          (df['decisionMake'] * 10) + (df[ 'touchQuality'] * 12.5) + (df['Aerial'] * 15)) / 10, 2)

################################################################################################################################################

def RoleForwardDefensivo(df):

    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['False 9'] =  round(((df['decisionMake'] * 20) + (df['positioning Midfield'] * 15) +
                            (df['touchQuality'] * 15) + (df['dribbles'] * 10) + (df['ballPlaying'] * 10) +
                            (df['attackingThreat'] * 30)) / 10, 2)

    ##########################################################################################################################################################################

    df['Target Man'] =  round(((df['decisionMake'] * 15) + (df['touchQuality'] * 10) +
                            (df['Aerial'] * 45) + (df['attackingThreat'] * 30)) / 10, 2)
    
    ##########################################################################################################################################################################

    df['Box Forward'] =  round(((df['decisionMake'] * 12.5) + (df['Aerial'] * 12.5) + (df['attackingThreat FW'] * 75)) / 10, 2)

    ##########################################################################################################################################################################

    df['Advanced Forward'] =  round(((df['ballPlaying'] * 10) + (df[ 'decisionMake'] * 15) +
                                    (df['touchQuality'] * 10) + (df[ 'dribbles'] * 10) +
                                    (df['Aerial'] * 20) + (df[ 'attackingThreat'] * 35)) / 10, 2)

################################################################################################################################################
#--------------------------------------------------- DIRECT --------------------------------------------------------------------------------
################################################################################################################################################

def RoleCenterBackDirect(df):

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10


    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Stopper'] = round(((df['Defensive Ability'] * 27.5) + (df['positioning Defence'] * 30) + (df[ 'Aerial'] * 20) +
                           (df['ballPlaying'] * 12.5) + (df['attackingThreat Defence'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Aerial CB'] =  round(((df['Defensive Ability'] * 15) + (df['positioning Defence'] * 20) + (df[ 'Aerial'] * 40) +
                              (df[ 'ballPlaying'] * 10) + (df[ 'progressiveRuns'] * 5) + (df['attackingThreat Defence'] * 10)) / 10, 2)
 
    ##########################################################################################################################################################################

    df['Ball Playing CB'] =  round(((df['Defensive Ability'] * 20) + (df['positioning Defence'] * 15) + (df[ 'Aerial'] * 5) +
                                    (df[ 'ballPlaying'] * 37.5) + (df[ 'progressiveRuns'] * 17.5) + (df['attackingThreat Defence'] * 5)) / 10, 2)

    ##########################################################################################################################################################################

    df['Ball Carrying CB'] =  round(((df['Defensive Ability'] * 22.5) + (df[ 'positioning Defence'] * 15) + (df[ 'Aerial'] * 10) +
                            (df[ 'ballPlaying'] * 12.5) + (df[ 'progressiveRuns'] * 35) + (df[ 'attackingThreat Defence'] * 5)) / 10, 2)

################################################################################################################################################

def roleFullBackDirect(df):

    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Attacking FB'] =  round(((df['crossing'] * 15) + (df['dribbles'] * 10) + (df['positioning Defence'] * 10) +
                                 (df['progressiveRuns'] * 20) + (df['defending1v1'] * 10) + (df['decisionMake'] * 15) + (df['touchQuality'] * 20)) / 10, 2)

    ##########################################################################################################################################################################

    df['Wing Back'] =  round(((df['crossing'] * 15) + (df[ 'dribbles'] * 10) + (df[ 'positioning Defence'] * 15) +
                              (df['progressiveRuns'] * 25) +(df[ 'decisionMake'] * 17.5) + (df[ 'touchQuality'] * 17.5)) / 10, 2)

    ##########################################################################################################################################################################

    df['Inverted Wing Back'] =  round(((df['dribbles'] * 15) + (df['defending1v1'] * 20) + (df['positioning Midfield'] * 10) +
                            (df['positioning Defence'] * 20) + (df['decisionMake'] * 20) + (df[ 'touchQuality'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Defensive FB'] =  round(((df['crossing'] * 10) + (df[ 'defending1v1'] * 35) +
                                (df['positioning Defence'] * 25) + (df['progressiveRuns'] * 10)  + (df['touchQuality'] * 10) + (df[ 'Aerial'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Full Back CB'] =  round(((df['crossing'] * 10) + (df[ 'defending1v1'] * 45) +
                                (df['positioning Defence'] * 35) + (df[ 'Aerial'] * 10)) / 10, 2)

################################################################################################################################################

def RoleMidfieldDirect(df):

    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df['rank_Forward passes/90'] * 25) + (df['rank_Progressive passes/90'] * 25) +
                            (df['rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df['rank_Forward passes %'] * 12.5) +
                              (df['rank_Forward passes/90'] * 22.5) + (df['rank_Progressive passes/90'] * 10) +
                              (df['rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Ball Winner'] =  round(((df['positioning Midfield'] * 20) + (df['defending1v1'] * 21) +
                                (df['touchQuality'] * 17) + (df['ballPlaying'] * 25) +
                                (df['Aerial'] * 17)) / 10, 2)

    ##########################################################################################################################################################################

    df['Deep Lying Playmaker'] =  round(((df['positioning Midfield'] * 12) + (df['decisionMake'] * 20) +
                                         (df['touchQuality'] * 30) + (df[ 'ballPlaying Deep'] * 38)) / 10, 2)

    ##########################################################################################################################################################################

    df['Attacking Playmaker'] =  round(((df['positioning Midfield'] * 10) + (df['progressiveRuns'] * 10) +
                                    (df['decisionMake'] * 25.5) + (df['touchQuality'] * 20.5) + (df['dribbles'] * 14) + (df['ballPlaying'] * 20)) / 10, 2)

    ##########################################################################################################################################################################

    df['Media Punta llegador'] =  round(((df['attackingThreat'] * 10) + (df['progressiveRuns'] * 14) +
                                    (df['decisionMake'] * 25.5) + (df['touchQuality'] * 16.5) + (df['dribbles'] * 14) + (df['ballPlaying'] * 20)) / 10, 2)

    ##########################################################################################################################################################################

    df['Box-to-box'] =  round(((df['Pass_Forward'] * 35) + (df['positioning Midfield'] * 35) + (df['runs'] * 30))/ 10, 2)

################################################################################################################################################

def RoleExtremoDirect(df):
    
    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Banda'] =  round(((df['progressiveRuns'] * 30) + (df['decisionMake'] * 10) +
                          (df['defending1v1'] * 5) + (df['dribbles'] * 15) + (df['crossing'] * 30) +
                          (df['attackingThreat'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Por dentro'] =  round(((df['progressiveRuns'] * 20) + (df['decisionMake'] * 17.5) +
                                (df['touchQuality'] * 20) + (df['dribbles'] * 27.5) +
                                (df['attackingThreat'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Defensivo'] =  round(((df['crossing'] * 20) + (df['dribbles'] * 10) + (df['positioning Midfield'] * 20) +
                              (df['decisionMake'] * 15) + (df[ 'touchQuality'] * 15) + (df['defending1v1'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Falso'] =  round(((df['attackingThreat'] * 40) + (df['dribbles'] * 10) +
                          (df['decisionMake'] * 15) + (df[ 'touchQuality'] * 15) + (df['Aerial'] * 10)) / 10, 2)

################################################################################################################################################

def RoleForwardDirect(df):

    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['False 9'] =  round(((df['decisionMake'] * 20) + (df['touchQuality'] * 25) + (df['dribbles'] * 10) + (df['ballPlaying'] * 15) +
                            (df['attackingThreat'] * 30)) / 10, 2)

    ##########################################################################################################################################################################

    df['Target Man'] =  round(((df['decisionMake'] * 17.5) + (df['touchQuality'] * 15) +
                            (df['Aerial'] * 40) + (df['attackingThreat'] * 30)) / 10, 2)
    
    ##########################################################################################################################################################################

    df['Box Forward'] =  round(((df['decisionMake'] * 12.5) + (df['Aerial'] * 12.5) + (df['attackingThreat FW'] * 75)) / 10, 2)

    ##########################################################################################################################################################################

    df['Advanced Forward'] =  round(((df['ballPlaying'] * 10) + (df['decisionMake'] * 20) +
                                    (df['touchQuality'] * 17.5) + (df['dribbles'] * 5) +
                                    (df['Aerial'] * 15) + (df[ 'attackingThreat'] * 32.5)) / 10, 2)
    
################################################################################################################################################
#--------------------------------------------------- POSSESSION --------------------------------------------------------------------------------
################################################################################################################################################

def RoleCenterBack(df):

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10


    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Stopper'] = round(((df['Defensive Ability'] * 27.5) + (df['positioning Defence'] * 30) + (df[ 'Aerial'] * 20) +
                           (df['ballPlaying'] * 12.5) + (df['attackingThreat Defence'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Aerial CB'] =  round(((df['Defensive Ability'] * 15) + (df['positioning Defence'] * 20) + (df[ 'Aerial'] * 40) +
                              (df[ 'ballPlaying'] * 10) + (df[ 'progressiveRuns'] * 5) + (df['attackingThreat Defence'] * 10)) / 10, 2)
 
    ##########################################################################################################################################################################

    df['Ball Playing CB'] =  round(((df['Defensive Ability'] * 20) + (df['positioning Defence'] * 15) + (df[ 'Aerial'] * 5) +
                                    (df[ 'ballPlaying'] * 37.5) + (df[ 'progressiveRuns'] * 17.5) + (df['attackingThreat Defence'] * 5)) / 10, 2)

    ##########################################################################################################################################################################

    df['Ball Carrying CB'] =  round(((df['Defensive Ability'] * 22.5) + (df[ 'positioning Defence'] * 15) + (df[ 'Aerial'] * 10) +
                            (df[ 'ballPlaying'] * 12.5) + (df[ 'progressiveRuns'] * 35) + (df[ 'attackingThreat Defence'] * 5)) / 10, 2)

################################################################################################################################################

def roleFullBack(df):

    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Attacking FB'] =  round(((df['crossing'] * 15) + (df['dribbles'] * 20) + (df['positioning Defence'] * 10) +
                                 (df['progressiveRuns'] * 20) + (df['defending1v1'] * 10) + (df['decisionMake'] * 12.5) + (df[ 'touchQuality'] * 12.5)) / 10, 2)

    ##########################################################################################################################################################################

    df['Wing Back'] =  round(((df['crossing'] * 20) + (df[ 'dribbles'] * 10) + (df[ 'positioning Defence'] * 15) +
                              (df[ 'progressiveRuns'] * 30) +(df[ 'decisionMake'] * 15) + (df[ 'touchQuality'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Inverted Wing Back'] =  round(((df['dribbles'] * 15) + (df['defending1v1'] * 20) + (df['positioning Midfield'] * 10) +
                            (df['positioning Defence'] * 20) + (df['decisionMake'] * 20) + (df[ 'touchQuality'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Defensive FB'] =  round(((df['crossing'] * 10) + (df[ 'defending1v1'] * 35) +
                                (df['positioning Defence'] * 25) + (df['progressiveRuns'] * 10)  + (df['touchQuality'] * 10) + (df[ 'Aerial'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Full Back CB'] =  round(((df['crossing'] * 10) + (df[ 'defending1v1'] * 45) +
                                (df['positioning Defence'] * 35) + (df[ 'Aerial'] * 10)) / 10, 2)

################################################################################################################################################

def RoleMidfield(df):

    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df['rank_Forward passes/90'] * 25) + (df['rank_Progressive passes/90'] * 25) +
                            (df['rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df['rank_Forward passes %'] * 12.5) +
                              (df['rank_Forward passes/90'] * 22.5) + (df['rank_Progressive passes/90'] * 10) +
                              (df['rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Ball Winner'] =  round(((df['positioning Midfield'] * 20) + (df['defending1v1'] * 21) +
                                (df['touchQuality'] * 17) + (df['ballPlaying'] * 25) +
                                (df['Aerial'] * 17)) / 10, 2)

    ##########################################################################################################################################################################

    df['Deep Lying Playmaker'] =  round(((df['positioning Midfield'] * 12) + (df['decisionMake'] * 20) +
                                         (df['touchQuality'] * 30) + (df[ 'ballPlaying Deep'] * 38)) / 10, 2)

    ##########################################################################################################################################################################

    df['Attacking Playmaker'] =  round(((df['positioning Midfield'] * 10) + (df['progressiveRuns'] * 10) +
                                    (df['decisionMake'] * 25.5) + (df['touchQuality'] * 20.5) + (df['dribbles'] * 14) + (df['ballPlaying'] * 20)) / 10, 2)

    ##########################################################################################################################################################################

    df['Media Punta llegador'] =  round(((df['attackingThreat'] * 10) + (df['progressiveRuns'] * 14) +
                                    (df['decisionMake'] * 25.5) + (df['touchQuality'] * 16.5) + (df['dribbles'] * 14) + (df['ballPlaying'] * 20)) / 10, 2)

    ##########################################################################################################################################################################

    df['Box-to-box'] =  round(((df['Pass_Forward'] * 35) + (df['positioning Midfield'] * 35) + (df['runs'] * 30))/ 10, 2)

################################################################################################################################################

def RoleExtremo(df):
    
    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['Banda'] =  round(((df['progressiveRuns'] * 30) + (df['decisionMake'] * 10) +
                          (df['defending1v1'] * 5) + (df['dribbles'] * 15) + (df['crossing'] * 30) +
                          (df['attackingThreat'] * 10)) / 10, 2)

    ##########################################################################################################################################################################

    df['Por dentro'] =  round(((df['progressiveRuns'] * 20) + (df['decisionMake'] * 17.5) +
                                (df['touchQuality'] * 17.5) + (df['dribbles'] * 30) +
                                (df['attackingThreat'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Defensivo'] =  round(((df['crossing'] * 20) + (df['dribbles'] * 10) + (df['positioning Midfield'] * 20) +
                              (df['decisionMake'] * 15) + (df[ 'touchQuality'] * 15) + (df['defending1v1'] * 15)) / 10, 2)

    ##########################################################################################################################################################################

    df['Falso'] =  round(((df['attackingThreat'] * 40) + (df['dribbles'] * 15) +
                          (df['decisionMake'] * 15) + (df[ 'touchQuality'] * 20) + (df['Aerial'] * 10)) / 10, 2)

################################################################################################################################################

def RoleForward(df):

    ##########################################################################################################################################################################

    df['Defensive Ability'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_Defensive duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['crossing'] = ((df['rank_Crosses/90'] * 30) + (df[ 'rank_Accurate crosses, %'] * 30) +
                     (df[ 'rank_Crosses to goalie box/90'] * 40)) / 10

    ##########################################################################################################################################################################

    df['defending1v1'] = ((df['rank_Defensive duels/90'] * 35) + (df[ 'rank_PAdj Sliding tackles'] * 65)) / 10

    ##########################################################################################################################################################################

    df['positioning Defence'] = ((df['rank_Shots blocked/90'] * 40) + (df[ 'rank_PAdj Interceptions'] * 60)) / 10

    ##########################################################################################################################################################################

    df['positioning Midfield'] = ((df['rank_Shots blocked/90'] * 30) + (df[ 'rank_PAdj Interceptions'] * 70)) / 10

    ##########################################################################################################################################################################

    df['Aerial'] = ((df['rank_Aerial duels/90'] * 35) + (df[ 'rank_Aerial duels %'] * 65)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 15) +
                            (df[ 'rank_Forward passes/90'] * 25) + (df[ 'rank_Progressive passes/90'] * 25) +
                            (df[ 'rank_Accurate progressive passes, %'] * 15)) / 10

    ##########################################################################################################################################################################

    df['ballPlaying Deep'] = ((df['rank_Passes %'] * 20) + (df[ 'rank_Forward passes %'] * 12.5) +
                              (df[ 'rank_Forward passes/90'] * 22.5) + (df[ 'rank_Progressive passes/90'] * 10) +
                              (df[ 'rank_Accurate progressive passes, %'] * 10) + (df['rank_Lateral passes/90'] * 10) +
                              (df['rank_Accurate lateral passes, %'] * 10)) / 10

    ##########################################################################################################################################################################

    df['dribbles'] = ((df['rank_Dribbles/90'] * 65) + (df[ 'rank_Successful dribbles %'] * 35)) / 10

    ##########################################################################################################################################################################

    df['progressiveRuns'] = ((df['rank_Progressive runs/90'] * 50) + (df[ 'rank_Accelerations/90'] * 20) +
                                (df[ 'rank_Dribbles/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['Pass_Forward'] =  ((df['rank_Forward passes %'] * 20) + (df['rank_Forward passes/90'] * 50) + (df['rank_Progressive passes/90'] * 30)) / 10

    ##########################################################################################################################################################################

    df['runs'] =  ((df['rank_Progressive runs/90'] * 35) + (df['rank_Accelerations/90'] * 65)) / 10

    ##########################################################################################################################################################################

    df['decisionMake'] = ((df['rank_Shot assists/90'] * 20) + (df[ 'rank_Passes penalty area %'] * 20) +
                            (df[ 'rank_Assists/90'] * 10) + (df[ 'rank_Second assists/90'] * 15) +
                            (df[ 'rank_xA/90'] * 15) + (df[ 'rank_Passes to penalty area/90'] * 10) + (df[ 'rank_Deep completions/90'] * 10)) / 10

    ##########################################################################################################################################################################

    df['touchQuality'] = ((df['rank_Key passes/90'] * 30) + (df[ 'rank_Accurate smart passes, %'] * 25) +
                            (df[ 'rank_Smart passes/90'] * 25) + (df[ 'rank_Fouls suffered/90'] * 20)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat'] = ((df['rank_xG/90'] * 25) + (df[ 'rank_Goals/90'] * 25) +
                                (df[ 'rank_Head goals/90'] * 12) + (df[ 'rank_Shots/90'] * 12) + (df[ 'rank_Shots on target, %'] * 14) +
                                (df[ 'rank_Goal conversion, %'] * 12)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat FW'] = ((df['rank_xG'] * 20) + (df[ 'rank_Goals'] * 80)) / 10

    ##########################################################################################################################################################################

    df['attackingThreat Defence'] = ((df[ 'rank_Head goals/90'] * 50) + (df[ 'rank_Head goals'] * 50)) / 10

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################

    df['False 9'] =  round(((df['progressiveRuns'] * 5) + (df['decisionMake'] * 20) +
                            (df['touchQuality'] * 20) + (df['dribbles'] * 10) + (df['ballPlaying'] * 15) +
                            (df['attackingThreat'] * 30)) / 10, 2)

    ##########################################################################################################################################################################

    df['Target Man'] =  round(((df['decisionMake'] * 17.5) + (df['touchQuality'] * 15) +
                            (df['Aerial'] * 40) + (df['attackingThreat'] * 30)) / 10, 2)
    
    ##########################################################################################################################################################################

    df['Box Forward'] =  round(((df['decisionMake'] * 12.5) + (df['Aerial'] * 12.5) + (df['attackingThreat FW'] * 75)) / 10, 2)

    ##########################################################################################################################################################################

    df['Advanced Forward'] =  round(((df['ballPlaying'] * 10) + (df[ 'decisionMake'] * 20) +
                                    (df['touchQuality'] * 12.5) + (df[ 'dribbles'] * 10) +
                                    (df['Aerial'] * 15) + (df[ 'attackingThreat'] * 32.5)) / 10, 2)

################################################################################################################################################

def scoutReport(playerName, team, setPieces, isDefensive, tier):

    playerAbility(df)

    playerClub = df.loc[(df['Player'] == playerName) & (df['Team'] == team) & (df['Season'] == '2021/22')]

    playerClub['Market value'] = playerClub['Market value'].astype(str)

    age = playerClub['Age'].unique()
    age = age.tolist()
    age = age[0]

    league = playerClub['Comp'].unique()
    league = league.tolist()
    league = league[0]

    country = playerClub['Birth country'].unique()
    country = country.tolist()
    country = country[0]
    if country == '0':
        country = playerClub['Passport country'].unique()
        country = country.tolist()
        country = country[0]
    if ',' in country:
        country = country.split(', ')[0]

    Market = playerClub['Market value'].unique()
    Market = Market.tolist()
    valor = Market[0]

    if len(str(valor)) == 6:
        valor = str(valor)[:3]
                
    elif len(str(valor)) == 7:
        if str(valor)[:2][1] != 0:
            valor = str(valor)[:2][0] + '.' + str(valor)[:2][1] + 'M'
            
    elif len(str(valor)) == 8:
        valor = str(valor)[:2] + 'M'

    elif len(str(valor)) == 9:
        valor = str(valor)[:3] + 'M'

    position = playerClub['Position'].unique()
    position = position.tolist()
    position = position[0]
    if ', ' in position:
        position = position.split(', ')[0]

    Contract = playerClub['Contract expires'].unique()
    Contract = Contract.tolist()
    contrato = Contract[0]

    Height = playerClub['Height'].unique()
    Height = Height.tolist()
    altura = str(Height[0])

    Foot = playerClub['Foot'].unique()
    Foot = Foot.tolist()
    pé = Foot[0]

    Minutes = playerClub['Minutes played'].unique()
    Minutes = Minutes.tolist()
    Minutes = str(Minutes[0])
    Minutes = int(Minutes)

    club = playerClub['Team'].unique()
    club = club.tolist()
    club = club[0]

    color = ['#FF0000', '#181818']

    #######################################################################################################################################

    fig = plt.figure(figsize=(15, 10), dpi=1000, facecolor = '#E8E8E8')
    gspec = gridspec.GridSpec(
    ncols=2, nrows=2, wspace = 0.5
    )

    ########################################################################################################################################################

    ax1 = plt.subplot(
                    gspec[0, 0],
            )


    ax1.axis('off')

    ax2 = plt.subplot(
                    gspec[1, 1],
            )

    pitch = Pitch(pitch_type='opta',pad_top=-0.5, pad_bottom=0.5, pad_right=-0.5,
                                pitch_color='#E8E8E8', line_color='#1b1b1b',line_zorder=1, linewidth=5, spot_scale=0.002)

    pitch.draw(ax=ax2)

    if 'GK' in position:
        # GK
        pitch.scatter(x=5.8, y=50, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

#######################################################################################################################################

    elif 'RB' in position:
        # RIGHT FULLBACK
        pitch.scatter(x=17, y=15, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

    elif 'RCB' in position:
        # RIGHT CENTER BACK
        pitch.scatter(x=17, y=36.8, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

    elif 'LCB' in position:
        # LEFT CENTER BACK
        pitch.scatter(x=17, y=63.2, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

    elif 'LB' in position:
        # LEFT FULLBACK
        pitch.scatter(x=17, y=85, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

#######################################################################################################################################

    elif 'DM' in position:
        # DEFENSIVE MIDFIELDER
        pitch.scatter(x=40, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

    elif 'RCMF' in position:
        # CENTER MIDFIELDER RIGHT
        pitch.scatter(x=55, y=36.8, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

    elif 'LCMF' in position:
        # CENTER MIDFIELDER LEFT
        pitch.scatter(x=40, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

#######################################################################################################################################
    elif 'RAMF' in position:
        # RIGHT MIDFIELDER
        pitch.scatter(x=70, y=15, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

    elif 'AMF' in position:
        # OFFENSIVE MIDFIELDER
        pitch.scatter(x=70, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

    elif 'LAMF' in position:
        # LEFT MIDFIELDER
        pitch.scatter(x=70, y=85, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

#######################################################################################################################################

    elif 'LW' in position:
        # WINGER LEFT
        pitch.scatter(x=85, y=88, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

    elif 'LWB' in position:
        # WINGER LEFT
        pitch.scatter(x=85, y=88, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

    elif 'RW' in position:
        # WINGER RIGHT
        pitch.scatter(x=85, y=12, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

#######################################################################################################################################

    elif 'CF' in position:
        # FORWARD
        pitch.scatter(x=88.5, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

######################################################## MUTIPLE POSITIONS - LW / RW AND OM ###############################################################################

    elif (position == 'LCB & RCB') | (position == 'RCB & LCB'):
        # LEFT CENTER BACK
        pitch.scatter(x=17, y=63.2, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)
        
        # RIGHT CENTER BACK
        pitch.scatter(x=17, y=36.8, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

    elif (position == 'LW & AMF') | (position == 'OM & LW'):
        # WINGER LEFT
        pitch.scatter(x=85, y=88, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

        # OFFENSIVE MIDFIELDER
        pitch.scatter(x=70, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

    elif (position == 'RW & AMF') | (position == 'AMF & RW'):
        # WINGER RIGHT
        pitch.scatter(x=85, y=12, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

        # OFFENSIVE MIDFIELDER
        pitch.scatter(x=70, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

######################################################## MUTIPLE POSITIONS - LM / RM AND OM ###############################################################################

    elif (position == 'LAMF & AMF') | (position == 'AMF & LAMF'):
        # LEFT MIDFIELDER
        pitch.scatter(x=70, y=85, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

        # OFFENSIVE MIDFIELDER
        pitch.scatter(x=70, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

    elif (position == 'RAMF & AMF') | (position == 'AMF & RAMF'):
        # RIGHT MIDFIELDER
        pitch.scatter(x=70, y=15, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

        # OFFENSIVE MIDFIELDER
        pitch.scatter(x=70, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

######################################################## MUTIPLE POSITIONS - AMF, CF, RCMF ###############################################################################

    elif (position == 'AMF, CF, RCMF'):
        # FORWARD
        pitch.scatter(x=88.5, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

        # OFFENSIVE MIDFIELDER
        pitch.scatter(x=70, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

        # RIGHT MIDFIELDER
        pitch.scatter(x=70, y=15, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)
######################################################## MUTIPLE POSITIONS - FW AND OM ###############################################################################

    elif (position == 'FW & AMF') | (position == 'AMF & FW'):
        # FORWARD
        pitch.scatter(x=88.5, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

        # OFFENSIVE MIDFIELDER
        pitch.scatter(x=70, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

######################################################## MUTIPLE POSITIONS - RB AND LB ###############################################################################

    elif (position == 'LB & RB') | (position == 'RB & LB'):
        # RIGHT FULLBACK
        pitch.scatter(x=17, y=15, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

        # LEFT FULLBACK
        pitch.scatter(x=17, y=85, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

######################################################## MUTIPLE POSITIONS - LCB AND LB ###############################################################################

    elif (position == 'LB & LCB') | (position == 'LCB & LB'):
        # LEFT CENTER BACK
        pitch.scatter(x=17, y=36.8, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)
        
        # LEFT FULLBACK
        pitch.scatter(x=17, y=15, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

######################################################## MUTIPLE POSITIONS - RCB AND RB ###############################################################################

    elif (position == 'RB & RCB') | (position == 'RCB & RB'):
        # RIGHT CENTER BACK
        pitch.scatter(x=17, y=63.2, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)
        
        # RIGHT FULLBACK
        pitch.scatter(x=17, y=15, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

######################################################## MUTIPLE POSITIONS - DM AND CML ###############################################################################

    elif (position == 'DMF & LDMF') | (position == 'LDMF & DMF'):
        # DEFENSIVE MIDFIELDER
        pitch.scatter(x=40, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)
        
        # CENTER MIDFIELDER LEFT
        pitch.scatter(x=55, y=36.8, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

######################################################## MUTIPLE POSITIONS - DM AND CMR ###############################################################################

    elif (position == 'RDMF & DMF') | (position == 'DMF, RDMF'):
        # DEFENSIVE MIDFIELDER
        pitch.scatter(x=40, y=50, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)
        
        # CENTER MIDFIELDER RIGHT
        pitch.scatter(x=55, y=36.8, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)

######################################################## MUTIPLE POSITIONS - CML AND CMR ###############################################################################

    elif (position == 'LCMF, RCMF') | (position == 'RCMF, LCMF'):
        # CENTER MIDFIELDER RIGHT
        pitch.scatter(x=55, y=36.8, ax=ax2, c=color[0], edgecolor='#1b1b1b', s=500, zorder=3)
        
        # CENTER MIDFIELDER LEFT
        pitch.scatter(x=55, y=63.2, ax=ax2, c=color[0], edgecolor="#1b1b1b", s=500, zorder=3)

#######################################################################################################################################

    cols_Offensive = ['Finishing Ability', 'Heading Ability', 'Concentration Ability', 'Sight play', 'Aerial Ability', 'Create Chances Ability', 'Dribbling Ability', 'SetPieces Ability', 'KeyPass Ability', 'Pass Ability']
    cols_Defensive = ['Heading Ability', 'Concentration Ability', 'Pass Ability', 'SetPieces Ability', 'Aerial Ability', 'Tackle Ability', 'Interception Ability']

    if (isDefensive == True) & (setPieces == False):
        params = cols_Defensive

        cols_Defensive.remove('SetPieces Ability')
    
        player = df.loc[(df['Player'] == playerName) & (df['Comp'] == league)][cols_Defensive].reset_index()
        player = list(player.loc[0])
        player = player[1:]

    elif (isDefensive == True) & (setPieces == True):

        params = cols_Defensive

        player = df.loc[(df['Player'] == playerName) & (df['Comp'] == league)][cols_Defensive].reset_index()
        player = list(player.loc[0])
        player = player[1:]

    elif (isDefensive == False) & (setPieces == True):

        params = cols_Offensive

        player = df.loc[(df['Player'] == playerName) & (df['Comp'] == league)][cols_Offensive].reset_index()
        player = list(player.loc[0])
        player = player[1:]

    elif (isDefensive == False) & (setPieces == False):

        params = cols_Offensive

        cols_Offensive.remove('SetPieces Ability')
    
        player = df.loc[(df['Player'] == playerName) & (df['Comp'] == league)][cols_Offensive].reset_index()
        player = list(player.loc[0])
        player = player[1:]

    #######################################################################################################################################

    df2 = df.loc[(df['Tier'] == tier) & (df['Minutes played'] >= Minutes) | (df.Player == playerName)].reset_index(drop=True)

    values = []
    for x in range(len(params)):   
        values.append(math.floor(stats.percentileofscore(df2[params[x]], player[x])))

    for n,i in enumerate(values):
        if i == 100:
            values[n] = 99
    
    elite = []
    veryStrong = []
    strong = []
    improve = []
    weak = []
    veryWeak = []
    for i in range(len(values)):
        if values[i] >= 95:
            elite.append(params[i])

        elif values[i] >= 80 | values[i] < 95:
            veryStrong.append(params[i])

        elif values[i] >= 70 | values[i] < 80:
            strong.append(params[i])

        elif values[i] >= 50 | values[i] < 70:
            improve.append(params[i])

        elif values[i] < 50:
            weak.append(params[i])

        elif values[i] <= 30:
            veryWeak.append(params[i])
        

    #######################################################################################################################################
    highlight_textpropsStrong =\
    [{"color": '#2ae102', "fontweight": 'bold'}]

    fig_text(x = 0.53, y = 0.9,
            s='Strengths',
            size=28,
            color='#2ae102')

    h=0.94
    for i in elite:
        ax_text(x=1.5, y=h - 0.1, s='<Elite:>' + ' ' + i, va='center', ha='center',
                highlight_textprops = highlight_textpropsStrong, size=18, color='#1b1b1b', ax=ax1)
        h=h-0.1

    h=h
    for i in veryStrong:
        ax_text(x=1.5, y=h - 0.1, s='<Very Strong:>' + ' ' + i, va='center', ha='center',
                highlight_textprops = highlight_textpropsStrong, size=18, color='#1b1b1b', ax=ax1)
        h=h-0.1

    h=h
    for i in strong:
        ax_text(x=1.5, y=h - 0.1, s='<Strong:>' + ' ' + i, va='center', ha='center',
                highlight_textprops = highlight_textpropsStrong, size=18, color='#1b1b1b', ax=ax1)
        h=h-0.1

    highlight_textprops =\
    [{"color": '#ff0000', "fontweight": 'bold'},
     {"color": '#f48515', "fontweight": 'bold'}]

    fig_text(x = 0.75, y = 0.9,
             s='<Weaknesses>/<To improve>',
             highlight_textprops = highlight_textprops,
             size=28)

    highlight_textpropsImprove =\
    [{"color": '#f48515', "fontweight": 'bold'}]

    highlight_textpropsWeak =\
    [{"color": '#ff0000', "fontweight": 'bold'}]

    h=0.94
    for i in improve:
        ax_text(x=2.5, y=h - 0.1, s='<Improve:>' + ' ' + i, va='center', ha='center',
                highlight_textprops = highlight_textpropsImprove, size=18, color='#1b1b1b', ax=ax1)
        h=h-0.1

    h=h
    for i in weak:
        ax_text(x=2.5, y=h - 0.1, s='<Weak:>' + ' ' + i, va='center', ha='center',
                highlight_textprops = highlight_textpropsWeak, size=18, color='#1b1b1b', ax=ax1)
        h=h-0.1

    h=h
    for i in veryWeak:
        ax_text(x=2.5, y=h - 0.1, s='<Very Weak:>' + ' ' + i, va='center', ha='center',
                highlight_textprops = highlight_textpropsWeak, size=18, color='#1b1b1b', ax=ax1)
        h=h-0.1
        
    #######################################################################################################################################

    fig_text(x = 0.23, y = 0.91,
            s=playerName,
            size=30,
            color='#1b1b1b')

    fig_text(x = 0.23, y = 0.75,
            s='Club: ',
            size=20,
            color='#1b1b1b')

    fig_text(x = 0.21, y = 0.71,
            s='League: ',
            size=20,
            color='#1b1b1b')

    fig_text(x = 0.16, y = 0.67,
            s='Market Value: ',
            size=20,
            color='#1b1b1b')

    fig_text(x = 0.15, y = 0.63,
            s='Contract until: ',
            size=20,
            color='#1b1b1b')

    fig_text(x = 0.15, y = 0.59,
            s='Minutes played: ',
            size=20,
            color='#1b1b1b')

    fig_text(x = 0.223, y = 0.55,
            s='Age: ',
            size=20,
            color='#1b1b1b')

    fig_text(x = 0.223, y = 0.51,
            s='Height: ',
            size=20,
            color='#1b1b1b')

    fig_text(x = 0.233, y = 0.47,
            s='Foot: ',
            size=20,
            color='#1b1b1b')

#######################################################################################################################################

    highlight_textprops =\
    [{"color": color[0], "fontweight": 'bold'}]

    fig_text(x = 0.3, y = 0.75,
            s=f'<{club}>',
            highlight_textprops = highlight_textprops,
            size=18,
            color='#1b1b1b')

    fig_text(x = 0.301, y = 0.71,
            s=league,
            size=18,
            color='#1b1b1b')

    fig_text(x = 0.301, y = 0.67,
            s=valor,
            size=18,
            color='#1b1b1b')

    #if 'M' not in Market:
    #    fig_text(x = 0.33, y = 0.667,
    #            s='Thousand',
    #            size=11,
    #            color='#1b1b1b')

    fig_text(x = 0.301, y = 0.63,
            s=contrato,
            size=18,
            color='#1b1b1b')

    fig_text(x = 0.301, y = 0.59,
            s=str(Minutes),
            size=18,
            color='#1b1b1b')

    fig_text(x = 0.301, y = 0.55,
            s=str(age),
            size=18,
            color='#1b1b1b')

    fig_text(x = 0.301, y = 0.51,
            s=altura,
            size=18,
            color='#1b1b1b')

    fig_text(x = 0.301, y = 0.47,
            s=pé,
            size=18,
            color='#1b1b1b')

#######################################################################################################################################

    fig_text(x = 0.47, y = 0.41,
            s='<Main Position>',
            highlight_textprops = highlight_textprops,
            size=18,
            color='#1b1b1b')

    if (position.split(', ')[0] == 'RAMF') | (position == 'RAMF'):
        fig_text(x = 0.423, y = 0.38,
                s='RIGHT MIDFIELDER',
                size=18,
                color='#1b1b1b')

    elif (position.split(', ')[0] == 'LAMF') | (position == 'LAMF'):
        fig_text(x = 0.423, y = 0.38,
                s='LEFT MIDFIELDER',
                size=18,
                color='#1b1b1b')
    
    elif (position.split(', ')[0] == 'CF') | (position == 'CF'):
        fig_text(x = 0.423, y = 0.38,
                s='FORWARD',
                size=18,
                color='#1b1b1b')
    
    elif (position.split(', ')[0] == 'RW') | (position == 'RW'):
        fig_text(x = 0.423, y = 0.38,
                s='RIGHT WINGER',
                size=18,
                color='#1b1b1b')

    elif (position.split(', ')[0] == 'LW') | (position == 'LW') | (position.split(', ')[0] == 'LWB') :
        fig_text(x = 0.423, y = 0.38,
                s='LEFT WINGER',
                size=18,
                color='#1b1b1b')

    elif (position.split(', ')[0] == 'LB') | (position == 'LB'):
        fig_text(x = 0.423, y = 0.38,
                s='LEFT BACK',
                size=18,
                color='#1b1b1b')
    
    elif (position.split(', ')[0] == 'RB') | (position == 'RB'):
        fig_text(x = 0.423, y = 0.38,
                s='RIGHT BACK',
                size=18,
                color='#1b1b1b')
    
    elif (position.split(', ')[0] == 'LCB') | (position == 'LCB'):
        fig_text(x = 0.423, y = 0.38,
                s='LEFT CENTER BACK',
                size=18,
                color='#1b1b1b')
    
    elif (position.split(', ')[0] == 'RCB') | (position == 'RCB'):
        fig_text(x = 0.423, y = 0.38,
                s='RIGHT CENTER BACK',
                size=18,
                color='#1b1b1b')

    elif (position.split(', ')[0] == 'DMF') | (position == 'DMF'):
        fig_text(x = 0.423, y = 0.38,
                s='DEFENSIVE MIDFIELDER',
                size=18,
                color='#1b1b1b')

    elif (position.split(', ')[0] == 'RCMF') | (position == 'RCMF'):
        fig_text(x = 0.423, y = 0.38,
                s='CENTER MIDFIELDER RIGHT',
                size=18,
                color='#1b1b1b')


    elif (position.split(', ')[0] == 'LCMF') | (position == 'LCMF'):
        fig_text(x = 0.423, y = 0.38,
                s='CENTER MIDFIELDER LEFT',
                size=18,
                color='#1b1b1b')

    elif (position.split(', ')[0] == 'AMF') | (position == 'AMF'):
        fig_text(x = 0.423, y = 0.38,
                s='OFFENSIVE MIDFIELDER',
                size=18,
                color='#1b1b1b')

    elif (position.split(' &')[0] == 'GK') | (position == 'GK'):
        fig_text(x = 0.423, y = 0.38,
                s='GOALKEEPER',
                size=18,
                color='#1b1b1b')


    #######################################################################################################################################
        
    if (position.__contains__('CB')):

        cols = ['Stopper', 'Aerial CB', 'Ball Playing CB', 'Ball Carrying CB']

        dfLeague = df.loc[df['Comp'] == league].reset_index(drop=True)

        RoleCenterBack(dfLeague)

        player = dfLeague.loc[(dfLeague['Position'].str.contains(position))].reset_index(drop=True)

        player = player.loc[(player['Player'] == playerName)][cols].reset_index()
        player = list(player.loc[0])
        player = player[1:]

        params = cols

        values = []
        for x in range(len(params)):   
            values.append(math.floor(stats.percentileofscore(dfLeague[params[x]], player[x])))

        for n,i in enumerate(values):
            if i == 100:
                values[n] = 99

    elif ((position.__contains__('RB')) | (position.__contains__('RWB')) | (position.__contains__('LWB')) | (position.__contains__('LB'))):

        cols = ['Inverted Wing Back', 'Wing Back', 'Attacking FB', 'Defensive FB', 'Full Back CB']

        dfLeague = df.loc[df['Comp'] == league].reset_index(drop=True)

        roleFullBack(dfLeague)

        player = dfLeague.loc[(dfLeague['Position'].str.contains(position))].reset_index(drop=True)

        player = player.loc[(player['Player'] == playerName)][cols].reset_index()
        player = list(player.loc[0])
        player = player[1:]

        params = cols

        values = []
        for x in range(len(params)):   
            values.append(math.floor(stats.percentileofscore(dfLeague[params[x]], player[x])))

        for n,i in enumerate(values):
            if i == 100:
                values[n] = 99


    elif ((position.__contains__('RCMF')) | (position.__contains__('LCMF')) | (position.__contains__('LDMF')) | (position.__contains__('RDMF')) | (position.__contains__('AMF')) | (position.__contains__('DMF'))):

        cols = ['Ball Winner', 'Deep Lying Playmaker', 'Box-to-box', 'Attacking Playmaker', 'Media Punta llegador']

        dfLeague = df.loc[df['Comp'] == league].reset_index(drop=True)

        RoleMidfield(dfLeague)

        player = dfLeague.loc[(dfLeague['Position'].str.contains(position))].reset_index(drop=True)

        player = player.loc[(player['Player'] == playerName)][cols].reset_index()
        player = list(player.loc[0])
        player = player[1:]

        params = cols

        values = []
        for x in range(len(params)):   
            values.append(math.floor(stats.percentileofscore(dfLeague[params[x]], player[x])))

        for n,i in enumerate(values):
            if i == 100:
                values[n] = 99


    elif ((position.__contains__('LW')) | (position.__contains__('RW')) | (position.__contains__('LAMF')) | (position.__contains__('RAMF')) | (position.__contains__('LWF')) | (position.__contains__('RWF'))):

        cols = ['Banda', 'Por dentro', 'Falso', 'Defensivo']

        dfLeague = df.loc[df['Comp'] == league].reset_index(drop=True)

        RoleExtremo(dfLeague)

        player = dfLeague.loc[(dfLeague['Position'].str.contains(position))].reset_index(drop=True)

        player = player.loc[(player['Player'] == playerName)][cols].reset_index()
        player = list(player.loc[0])
        player = player[1:]

        params = cols

        values = []
        for x in range(len(params)):   
            values.append(math.floor(stats.percentileofscore(dfLeague[params[x]], player[x])))

        for n,i in enumerate(values):
            if i == 100:
                values[n] = 99


    elif (position.__contains__('CF')):

        cols = ['Box Forward', 'Advanced Forward', 'Target Man', 'False 9']

        dfLeague = df.loc[df['Comp'] == league].reset_index(drop=True)

        RoleForward(dfLeague)

        player = dfLeague.loc[(dfLeague['Position'].str.contains(position))].reset_index(drop=True)

        player = player.loc[(player['Player'] == playerName)][cols].reset_index()
        player = list(player.loc[0])
        player = player[1:]

        params = cols

        values = []
        for x in range(len(params)):   
            values.append(math.floor(stats.percentileofscore(dfLeague[params[x]], player[x])))

        for n,i in enumerate(values):
            if i == 100:
                values[n] = 99

    # PLAYER ROLE   
    #playerRole.index[0]

    fig_text(x = 0.17, y = 0.4,
            s='Final Rating',
            size=16,
            color='#1b1b1b')

    if max(values) >= 90:
        fig_text(x = 0.25, y = 0.38,
                s='A',
                size=100,
                color='#2ae102')

    elif (max(values) >= 65) & (max(values) < 90):
        fig_text(x = 0.25, y = 0.38,
                s='B',
                size=100,
                color='#d7ee1a')

    elif (max(values) >= 50) & (max(values) < 65):
        fig_text(x = 0.25, y = 0.38,
                s='C',
                size=100,
                color='#fdab16')

    elif max(values) < 50:
        fig_text(x = 0.25, y = 0.38,
                s='D',
                size=100,
                color='#ff0000')

    #fig = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Players/' + league + '/' + club + '/' + playerName + '.png', fig=fig, left=0.12, bottom=0.78, width=0.1, height=0.23)

    #fig = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Country/' + country + '.png', fig=fig, left=0.23, bottom=0.775, width=0.1, height=0.07)


    plt.savefig('assets/scoutReport' + playerName + '.png', dpi=300)

    return app.get_asset_url('scoutReport' + playerName + '.png')
################################################################################################################################################


################################################################################################################################################
#--------------------------------------------------- ROLES --------------------------------------------------------------------------------
################################################################################################################################################

def szcore_df(data):

    cols_cat = []
    for col in data.columns:
        if col not in data.select_dtypes([np.number]).columns:
            cols_cat.append(col)

    for i in ['Age', 'Market value', 'Matches played', 'Minutes played', 'Height', 'Weight']:
        cols_cat.append(i)

    cols_num = [] 
    for col in data.select_dtypes([np.number]).columns:
            cols_num.append(col)
    
    for i in ['Age', 'Market value', 'Matches played', 'Minutes played', 'Height', 'Weight']:
        cols_num.remove(i)
        
    data2 = data[cols_cat]

    test = stats.zscore(data[cols_num])

    data3 = pd.concat([data2, test], axis=1)

    if 'Team within selected timeframe' not in data3.columns:
        pass
    else:
        data3.drop(['Team within selected timeframe', 'On loan'], axis=1, inplace=True)

    return data3


def bestRoleCB(data):
    
    data['Role'] = data[['Stopper', 'Aerial CB', 'Ball Playing CB', 'Ball Carrying CB']].T.apply(lambda x: x.nlargest(1).idxmin())

    data['Role2'] = data[['Stopper', 'Aerial CB', 'Ball Playing CB', 'Ball Carrying CB']].T.apply(lambda x: x.nlargest(2).idxmin())

    return data

################################################################################################################################################

def bestRoleFB(data):
    
    data['Role'] = data[['Inverted Wing Back', 'Wing Back', 'Attacking FB', 'Defensive FB', 'Full Back CB']].T.apply(lambda x: x.nlargest(1).idxmin())

    data['Role2'] = data[['Inverted Wing Back', 'Wing Back', 'Attacking FB', 'Defensive FB', 'Full Back CB']].T.apply(lambda x: x.nlargest(2).idxmin())

    return data

################################################################################################################################################

def bestRoleMidfield(data):
    
    data['Role'] = data[['Ball Winner', 'Deep Lying Playmaker', 'Box-to-box', 'Attacking Playmaker', 'Media Punta llegador']].T.apply(lambda x: x.nlargest(1).idxmin())

    data['Role2'] = data[['Ball Winner', 'Deep Lying Playmaker', 'Box-to-box', 'Attacking Playmaker', 'Media Punta llegador']].T.apply(lambda x: x.nlargest(2).idxmin())

    return data

################################################################################################################################################

def bestRoleExtremo(data):
    
    data['Role'] = data[['Banda', 'Por dentro', 'Falso', 'Defensivo']].T.apply(lambda x: x.nlargest(1).idxmin())

    data['Role2'] = data[['Banda', 'Por dentro', 'Falso', 'Defensivo']].T.apply(lambda x: x.nlargest(2).idxmin())

    return data

################################################################################################################################################

def bestRoleForward(data):
    
    data['Role'] = data[['Box Forward', 'Advanced Forward', 'Target Man', 'False 9']].T.apply(lambda x: x.nlargest(1).idxmin())

    data['Role2'] = data[['Box Forward', 'Advanced Forward', 'Target Man', 'False 9']].T.apply(lambda x: x.nlargest(2).idxmin())

    return data

################################################################################################################################################

def rangoCB(league, Role):
    
    dfCB = df.loc[df.Position.str.contains('CB')].reset_index(drop=True)

    playerAbility(dfCB)

    RoleCenterBack(dfCB)

    szcore_df(dfCB)

    bestRoleCB(dfCB)

    test = dfCB.loc[(dfCB['Comp'] == league) & (dfCB['Minutes played'] >= 1500)]

    role = Role

    dfRole = test.loc[((test['Role'] == role) | (test['Role2'] == role)) & ((test[role] >= test[role].quantile(.25)) | (test[role] <= test[role].quantile(.75)))].sort_values(role, ascending=False)
    
    return dfRole[['Player', 'Team', 'Comp', 'Role', 'Role2']]

################################################################################################################################################

def rangoFB(league, Role):
    
    dfFB = df.loc[(df.Position.str.contains('RB')) | (df.Position.str.contains('LB'))| (df.Position.str.contains('LWB'))| (df.Position.str.contains('RWB'))].reset_index(drop=True)

    playerAbility(dfFB)

    roleFullBack(dfFB)

    szcore_df(dfFB)

    bestRoleFB(dfFB)

    dfRole = dfFB.loc[(dfFB['Comp'] == league) & (dfFB['Minutes played'] >= 1000)]

    role = Role

    dfRole = dfRole.loc[((dfRole['Role'] == role) | (dfRole['Role2'] == role)) & ((dfRole[role] >= dfRole[role].quantile(.25)) | (dfRole[role] <= dfRole[role].quantile(.75)))].sort_values(role, ascending=False)
    
    return dfRole[['Player', 'Team', 'Comp', 'Role', 'Role2']]

################################################################################################################################################

def rangoMF(league, Role):
    
    dfMF = df.loc[(df.Position.str.contains('RCMF')) | (df.Position.str.contains('LCMF')) | (df.Position.str.contains('LDMF')) | (df.Position.str.contains('RDMF')) |
                    (df.Position.str.contains('AMF')) | (df.Position.str.contains('DMF'))].reset_index(drop=True)

    playerAbility(dfMF)

    RoleMidfield(dfMF)

    szcore_df(dfMF)

    bestRoleMidfield(dfMF)

    dfRole = dfMF.loc[(dfMF['Comp'] == league) & (dfMF['Minutes played'] >= 1500)]

    role = Role

    dfRole = dfRole.loc[((dfRole['Role'] == role) | (dfRole['Role2'] == role)) & ((dfRole[role] >= dfRole[role].quantile(.25)) | (dfRole[role] <= dfRole[role].quantile(.75)))].sort_values(role, ascending=False)

    return dfRole[['Player', 'Team', 'Comp', 'Role', 'Role2']]

################################################################################################################################################

def rangoExtremo(league, Role):
    
    dfExtremo = df.loc[(df.Position.str.contains('LW')) | (df.Position.str.contains('RW')) | (df.Position.str.contains('LAMF')) |
                       (df.Position.str.contains('RAMF')) | (df.Position.str.contains('LWF')) | (df.Position.str.contains('RWF'))].reset_index(drop=True)

    RoleExtremo(dfExtremo)

    szcore_df(dfExtremo)

    bestRoleExtremo(dfExtremo)

    dfRole = dfExtremo.loc[(dfExtremo['Comp'] == league) & (dfExtremo['Minutes played'] >= 1000)]

    role = Role

    dfRole = dfRole.loc[((dfRole['Role'] == role) | (dfRole['Role2'] == role)) & ((dfRole[role] >= dfRole[role].quantile(.25)) | (dfRole[role] <= dfRole[role].quantile(.75)))].sort_values(role, ascending=False)

    return dfRole[['Player', 'Team', 'Comp', 'Role', 'Role2']]

################################################################################################################################################

def rangoFW(league, role):
    
    dfFW = df.loc[(df.Position.str.contains('CF')) & (df['Season'] == '2021/22')].reset_index(drop=True)

    RoleForward(dfFW)

    szcore_df(dfFW)

    bestRoleForward(dfFW)

    dfRole = dfFW.loc[(dfFW['Comp'] == league) & (dfFW['Minutes played'] >= 1000)]

    dataRole = dfRole.loc[((dfRole['Role'] == role) | (dfRole['Role2'] == role)) & ((dfRole[role] >= dfRole[role].quantile(.25)) | (dfRole[role] <= dfRole[role].quantile(.75)))].sort_values(role, ascending=False)

    return dataRole[['Player', 'Team', 'Comp', 'Role', 'Role2']]

################################################################################################################################################

def dataFrameCenterBack(contract, tier, role, league, tier2=None):

    rango = rangoCB(league, role)

    rango = rango.tail(1)

    playeRange = rango.Player.values
    playeRange = playeRange.tolist()
    playeRange = playeRange[0]

    df = df.loc[(df.Position.str.contains('CB'))]

    playerAbility(df)

    RoleCenterBack(df)

    szcore_df(df)

    bestRoleCB(df)

    dfPlayer = df.loc[df.Player == playeRange].reset_index(drop=True)

    rangoValue = dfPlayer[role].values
    rangoValue = rangoValue.tolist()
    rangoValue = rangoValue[0]
    rangoValue

    def tierValue(role, tier):

       if tier == 0:
          return role * 1

       elif tier == 1:
          return role * 0.9

       elif tier == 2:
          return role * 0.8

       elif tier == 3:
          return role * 0.7

       elif tier == 4:
          return role * 0.65

    df[role] = df.apply(lambda x: tierValue(x[role], x['Tier']), axis=1)

    if tier != 'All Data':
        if contract == 'Yes':
            df = df.loc[ ((df.Tier == tier) | (df.Tier == tier2)) & (df['Minutes played'] >= 1500) & (df['Contract expires'] == str(date.today().year) + '-06' + '-30') & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
        
        elif contract == 'No':
            df = df.loc[((df.Tier == tier) | (df.Tier == tier2)) & (df['Minutes played'] >= 1500) & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
            
    elif tier == 'All Data':
        if contract == 'Yes':
            df = df.loc[(df['Contract expires'] == str(date.today().year) + '-06' + '-30') & (df['Minutes played'] >= 1500) & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
        
        elif contract == 'No':
            df = df.loc[(df[role] >= rangoValue) & (df['Minutes played'] >= 1500) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
            
    return df

################################################################################################################################################

def dataFrameFullBack(contract, tier, role, league, tier2=None):

    rango = rangoFB(league, role)

    rango = rango.tail(1)

    playeRange = rango.Player.values
    playeRange = playeRange.tolist()
    playeRange = playeRange[0]

    df = df.loc[((df.Position.str.contains('RB')) | (df.Position.str.contains('RWB')) | (df.Position.str.contains('LWB')) | (df.Position.str.contains('LB')))]


    playerAbility(df)

    roleFullBack(df)

    szcore_df(df)

    bestRoleFB(df)

    dfPlayer = df.loc[df.Player == playeRange].reset_index(drop=True)

    rangoValue = dfPlayer[role].values
    rangoValue = rangoValue.tolist()
    rangoValue = rangoValue[0]
    rangoValue

    def tierValue(role, tier):

       if tier == 0:
          return role * 1

       elif tier == 1:
          return role * 0.9

       elif tier == 2:
          return role * 0.8

       elif tier == 3:
          return role * 0.7

       elif tier == 4:
          return role * 0.65

    df[role] = df.apply(lambda x: tierValue(x[role], x['Tier']), axis=1)

    if tier != 'All Data':
        if contract == 'Yes':
            df = df.loc[ ((df.Tier == tier) | (df.Tier == tier2)) & (df['Minutes played'] >= 1500) & (df['Contract expires'] == str(date.today().year) + '-06' + '-30') & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
        
        elif contract == 'No':
            df = df.loc[((df.Tier == tier) | (df.Tier == tier2)) & (df['Minutes played'] >= 1500) & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
            
    elif tier == 'All Data':
        if contract == 'Yes':
            df = df.loc[(df['Contract expires'] == str(date.today().year) + '-06' + '-30') & (df['Minutes played'] >= 1500) & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
        
        elif contract == 'No':
            df = df.loc[(df[role] >= rangoValue) & (df['Minutes played'] >= 1500) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
            
    return df

################################################################################################################################################

def dataFrameMidfield(contract, tier, role, league, tier2=None):

    rango = rangoMF(league, role)

    rango = rango.tail(1)

    playeRange = rango.Player.values
    playeRange = playeRange.tolist()
    playeRange = playeRange[0]

    df = df.loc[((df.Position.str.contains('RCMF')) | (df.Position.str.contains('LCMF')) | (df.Position.str.contains('LDMF')) | (df.Position.str.contains('RDMF')) |
                    (df.Position.str.contains('AMF')) | (df.Position.str.contains('DMF')) | (df.Position.str.contains('RAMF')))].reset_index(drop=True)
                    
    playerAbility(df)

    RoleMidfield(df)

    szcore_df(df)

    bestRoleMidfield(df)

    def tierValue(role, tier):

       if tier == 0:
          return role * 1

       elif tier == 1:
          return role * 0.9

       elif tier == 2:
          return role * 0.8

       elif tier == 3:
          return role * 0.7

       elif tier == 4:
          return role * 0.65

    df[role] = df.apply(lambda x: tierValue(x[role], x['Tier']), axis=1)

    dfPlayer = df.loc[df.Player == playeRange].reset_index(drop=True)

    rangoValue = dfPlayer[role].values
    rangoValue = rangoValue.tolist()
    rangoValue = rangoValue[0]
    rangoValue

    if tier != 'All Data':
        if contract == 'Yes':
            df = df.loc[ ((df.Tier == tier) | (df.Tier == tier2)) & (df['Contract expires'] == str(date.today().year) + '-06' + '-30') & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
        
        elif contract == 'No':
            df = df.loc[((df.Tier == tier) | (df.Tier == tier2)) & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
            
    elif tier == 'All Data':
        if contract == 'Yes':
            df = df.loc[(df['Contract expires'] == str(date.today().year) + '-06' + '-30') & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
        
        elif contract == 'No':
            df = df.loc[(df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
            
    return df

################################################################################################################################################

def dataFrameExtremo(contract, tier, role, league, tier2=None):

    rango = rangoExtremo(league, role)

    rango = rango.tail(1)

    playeRange = rango.Player.values
    playeRange = playeRange.tolist()
    playeRange = playeRange[0]

    df = df.loc[(df.Position.str.contains('LW')) | (df.Position.str.contains('RW')) | (df.Position.str.contains('LAMF')) |
                     (df.Position.str.contains('RAMF')) | (df.Position.str.contains('LWF')) | (df.Position.str.contains('RWF'))]

    playerAbility(df)

    RoleExtremo(df)

    szcore_df(df)

    bestRoleExtremo(df)

    dfPlayer = df.loc[df.Player == playeRange].reset_index(drop=True)

    rangoValue = dfPlayer[role].values
    rangoValue = rangoValue.tolist()
    rangoValue = rangoValue[0]
    rangoValue

    def tierValue(role, tier):

       if tier == 0:
          return role * 1

       elif tier == 1:
          return role * 0.9

       elif tier == 2:
          return role * 0.8

       elif tier == 3:
          return role * 0.7

       elif tier == 4:
          return role * 0.65

    df[role] = df.apply(lambda x: tierValue(x[role], x['Tier']), axis=1)

    if tier != 'All Data':
        if contract == 'Yes':
            df = df.loc[ ((df.Tier == tier) | (df.Tier == tier2)) & (df['Minutes played'] >= 1500) & (df['Contract expires'] == str(date.today().year) + '-06' + '-30') & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
        
        elif contract == 'No':
            df = df.loc[((df.Tier == tier) | (df.Tier == tier2)) & (df['Minutes played'] >= 1500) & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
            
    elif tier == 'All Data':
        if contract == 'Yes':
            df = df.loc[(df['Contract expires'] == str(date.today().year) + '-06' + '-30') & (df['Minutes played'] >= 1500) & (df[role] >= rangoValue) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
        
        elif contract == 'No':
            df = df.loc[(df[role] >= rangoValue) & (df['Minutes played'] >= 1500) & ((df['Role'] == role) | (df['Role2'] == role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role, 'Tier']].sort_values(role, ascending=False)
            
    return df

################################################################################################################################################

def dataFrameForward(contract, tier, Role, League, tier2=None):

    playerAbility(df)

    rango = rangoFW(League, Role)

    rango = rango.tail(1)

    playeRange = rango.Player.values
    playeRange = playeRange.tolist()
    playeRange = playeRange[0]

    dfFW = df.loc[(df.Position.str.contains('CF')) & (df['Season'] == '2021/22')]

    #playerAbility(dfFW)

    RoleForward(dfFW)

    szcore_df(dfFW)

    bestRoleForward(dfFW)
    
    dfPlayer = dfFW.loc[dfFW.Player == playeRange].reset_index(drop=True)

    rangoValue = dfPlayer[Role].values
    rangoValue = rangoValue.tolist()
    rangoValue = rangoValue[0]
    rangoValue

    def tierValue(role, tier):

       if tier == 0:
          return role * 1

       elif tier == 1:
          return role * 0.9

       elif tier == 2:
          return role * 0.8

       elif tier == 3:
          return role * 0.7

       elif tier == 4:
          return role * 0.65
  
    dfFW[Role] = dfFW.apply(lambda x: tierValue(x[Role], x['Tier']), axis=1)

    if tier != 'All Data':
        if contract == 'Yes':
            dfFW = dfFW.loc[ ((dfFW.Tier == tier) | (dfFW.Tier == tier2)) & (dfFW['Minutes played'] >= 1500) & (dfFW['Contract expires'] == str(date.today().year) + '-06' + '-30') & (dfFW[Role] >= rangoValue) & ((dfFW['Role'] == Role) | (dfFW['Role2'] == Role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', Role, 'Tier']].sort_values(Role, ascending=False)
        
        elif contract == 'No':
            dfFW = dfFW.loc[((dfFW.Tier == tier) | (dfFW.Tier == tier2)) & (dfFW['Minutes played'] >= 1500) & (dfFW[Role] >= rangoValue) & ((dfFW['Role'] == Role) | (dfFW['Role2'] == Role))][['Player', 'Team', 'Passport country', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', Role, 'Tier']].sort_values(Role, ascending=False)
            
    elif tier == 'All Data':
        if contract == 'Yes':
            dfFW = dfFW.loc[(dfFW['Contract expires'] == str(date.today().year) + '-06' + '-30') & (dfFW['Minutes played'] >= 1500) & (dfFW[Role] >= rangoValue) & ((dfFW['Role'] == Role) | (dfFW['Role2'] == Role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', Role, 'Tier']].sort_values(Role, ascending=False)
        
        elif contract == 'No':
            dfFW = dfFW.loc[(dfFW[Role] >= rangoValue) & (dfFW['Minutes played'] >= 1500) & ((dfFW['Role'] == Role) | (dfFW['Role2'] == Role))][['Player', 'Team', 'Position', 'Status', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', Role, 'Tier']].sort_values(Role, ascending=False)
            
    return dfFW

################################################################################################################################################
#------------------------------------------------ TABLES ----------------------------------------------------------------------
################################################################################################################################################

def pathColumn(tableData):
        flag = []
        for idx, row in tableData.iterrows():
                if ', ' in tableData.loc[idx, 'Passport country']:
                        tableData.loc[idx, 'Passport country'] = tableData.loc[idx, 'Passport country'].split(', ')[0]
                        flag.append('C:/Users/menes/Documents/Data Hub/Country/' + tableData.loc[idx, 'Passport country'] + '.png')
                else:
                        flag.append('C:/Users/menes/Documents/Data Hub/Country/' + tableData.loc[idx, 'Passport country'] + '.png')
        
        return flag

################################################################################################################################################

def table(boxForward, role):
    
    boxForward['Flag'] = pathColumn(boxForward)
    
    boxForward = boxForward[['Player', 'Flag', 'Team', 'Position', 'Age', 'Comp', 'Contract expires', 'Market value', 'Role', 'Role2', role]].reset_index(drop=True)

    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                            ['#E8E8E8', '#3d0000', '#FF0000'], N=10)
    col_defs = (
        [
            ColumnDefinition(
                name="Player",
                textprops={"ha": "left", "weight": "bold"},
                width=1.2,
            ),
            ColumnDefinition(
                    name="Flag",
                    title="",
                    textprops={"ha": "center"},
                    width=0.5,
                    plot_fn=circled_image,
                ),
            ColumnDefinition(
                name="Team",
                textprops={"ha": "center"},
                width=1.5,
            ),
            ColumnDefinition(
                name="Position",
                textprops={"ha": "center"},
                width=0.75,
            ),
            ColumnDefinition(
                name="Age",
                textprops={"ha": "center"},
                width=0.75,
            ),
            ColumnDefinition(
                name="Comp",
                textprops={"ha": "center"},
                width=0.75,
            ),
            ColumnDefinition(
                name="Contract expires",
                textprops={"ha": "center"},
                width=0.75,
                group="Transfer data",
                border='left',
                plot_kw={"facecolor": "#E8E8E8"},
            ),
            ColumnDefinition(
                name="Market value",
                textprops={"ha": "center"},
                width=0.75,
                group="Transfer data",
                border='right',
                plot_kw={"facecolor": "#E8E8E8"},
            ),
            ColumnDefinition(
                name="Role",
                textprops={"ha": "center"},
                width=0.75,
                group="Role data",
                border='left',
            ),
            ColumnDefinition(
                name="Role2",
                textprops={"ha": "center"},
                width=0.75,
                group="Role data",
                border='right'
            ),
            ColumnDefinition(
                name=role,
                textprops={"ha": "center"},
                width=0.75,
                cmap=normed_cmap(boxForward[role], cmap=pearl_earring_cmap, num_stds=2.5),
                group="Role data",
            )
        ]
    )

    fig, ax = plt.subplots(figsize=(28, 35))

    fig.set_facecolor('white')

    table = Table(
        boxForward.set_index('Player').head(15),
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        textprops={"fontsize": 19},
        #cell_kw={'facecolor' : '#E8E8E8'},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 3, "linestyle": "-"},
        column_border_kw={"linewidth": 3, "linestyle": "-"},
    ).autoset_fontcolors(colnames=[role])

    #table.col_label_row.set_facecolor("#E8E8E8")


    fig_text(s = 'Best box forward players',
                    x = 0.52, y = 0.9, fontweight='bold',
                    ha='center',fontsize=65, color='#181818');

    fig_text(s = 'Only players at the big 5 league | Season 2021-22',
                    x = 0.5, y = 0.87, fontweight='bold',
                    ha='center',fontsize=16, color='#181818', alpha=0.8);

    plt.savefig('assets/table' + role + '.png', dpi=300)

    return app.get_asset_url('table' + role + '.png')

################################################################################################################################################















































































































