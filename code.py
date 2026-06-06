import json, os, pandas as pd
from pathlib import Path

def parse_cricsheet_match(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    info = data['info']
   
    if info.get('competition','').upper() not in ['IPL','INDIAN PREMIER LEAGUE']:
        competition = info.get('event', {}).get('name', '')
        if 'Premier League' not in competition and 'IPL' not in competition:
            return []
    
    season = info.get('season', 
             str(info.get('dates',[''])[0])[:4])
    venue  = info.get('venue', 'Unknown')
    teams  = info.get('teams', [])
    
    rows = []
    for innings in data.get('innings', []):
        batting_team = innings.get('team')
        
        for over_data in innings.get('overs', []):
            over_num = over_data['over']
            
            for delivery in over_data.get('deliveries', []):
                batter  = delivery['batter']
                bowler  = delivery['bowler']
                runs    = delivery['runs']['batter']
                extras  = delivery['runs']['extras']
                
                wicket = delivery.get('wickets', [])
                is_out = 1 if wicket else 0
                
                rows.append({
                    'season':       season,
                    'venue':        venue,
                    'batting_team': batting_team,
                    'batter':       batter,
                    'bowler':       bowler,
                    'runs_scored':  runs,
                    'extras':       extras,
                    'is_wicket':    is_out,
                    'over':         over_num,
                    'phase': 'powerplay' if over_num < 6 
                             else 'middle' if over_num < 15 
                             else 'death'
                })
    return rows


json_folder = Path('.')   
all_rows = []

json_files = list(json_folder.glob('*.json'))
print(f"Processing {len(json_files)} match files...")

for i, fpath in enumerate(json_files):
    try:
        rows = parse_cricsheet_match(fpath)
        all_rows.extend(rows)
    except Exception as e:
        print(f"Skipped {fpath.name}: {e}")
    
    if i % 100 == 0:
        print(f"  {i}/{len(json_files)} done...")

df = pd.DataFrame(all_rows)
df.to_csv('data/processed/cricsheet_deliveries.csv', index=False)
print(f"Done. Total deliveries: {len(df)}")



df = pd.read_csv('data/processed/cricsheet_deliveries.csv', low_memory=False)


df['season'] = df['season'].astype(str).str[:4].astype(int)
# ── Batting career stats ──
batting = df.groupby(['batter', 'season']).agg(
    runs        = ('runs_scored', 'sum'),
    balls_faced = ('runs_scored', 'count'),
    dismissals  = ('is_wicket', 'sum'),
 
    pp_runs  = ('runs_scored', lambda x: 
                x[df.loc[x.index,'phase']=='powerplay'].sum()),
    death_runs = ('runs_scored', lambda x:
                  x[df.loc[x.index,'phase']=='death'].sum()),
).reset_index()

batting['average']     = batting.runs / batting.dismissals.clip(1)
batting['strike_rate'] = (batting.runs / batting.balls_faced) * 100

# ── Bowling career stats ──
bowling = df.groupby(['bowler', 'season']).agg(
    wickets      = ('is_wicket', 'sum'),
    runs_conceded= ('runs_scored', 'sum'),
    balls_bowled = ('runs_scored', 'count'),
).reset_index()

bowling['economy'] = (bowling.runs_conceded / 
                      bowling.balls_bowled) * 6
bowling['bowling_avg'] = (bowling.runs_conceded / 
                           bowling.wickets.clip(1))

batting.to_csv('data/processed/batting_career.csv', index=False)
bowling.to_csv('data/processed/bowling_career.csv', index=False)
print("Batting players:", batting.batter.nunique())
print("Bowling players:", bowling.bowler.nunique())
