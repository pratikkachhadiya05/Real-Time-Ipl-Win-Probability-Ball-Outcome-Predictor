from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="IPL Win Predictor")

teams = [
    'Royal Challengers Bengaluru',
    'Punjab Kings',
    'Mumbai Indians',
    'Kolkata Knight Riders',
    'Chennai Super Kings',
    'Lucknow Super Giants',
    'Rajasthan Royals',
    'Gujarat Titans',
    'Sunrisers Hyderabad',
    'Delhi Capitals'
]

city_name = [
    'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Chennai', 'Delhi',
    'Dharamsala', 'Hyderabad', 'Jaipur', 'Kolkata', 'Lucknow',
    'Mohali', 'Mumbai', 'Nagpur', 'Pune', 'Raipur', 'Ranchi', 'Visakhapatnam'
]

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Load Models ────────────────────────────────────────────────────────────────
try:
    with open('pipe.pkl', 'rb') as f:
        model = pickle.load(f)
        print("✅ Win probability pipeline loaded successfully")
except FileNotFoundError:
    print("Error: 'pipe.pkl' not found.")
    model = None

try:
    with open('ball_model.pkl', 'rb') as f:
        model2 = pickle.load(f)
        print("✅ Ball prediction pipeline loaded successfully")
except FileNotFoundError:
    print("Error: 'ball_model.pkl' not found.")
    model2 = None


# ─── Pydantic Models ─────────────────────────────────────────────────────────────
class MatchData(BaseModel):
    batting_team: str
    bowling_team: str
    city: str
    target: int
    score: int
    overs: float
    wickets: int


class BallMatchData(BaseModel):
    """Extended match state that also carries ball-level context features."""
    batting_team: str
    bowling_team: str
    city: str
    target: int
    score: int
    overs: float
    wickets: int
    # add features for ball prediction
    last_5_balls_runs: int = 0
    last_18_balls_runs: int = 0
    partnership_runs: int = 0
    dots_last_6: int = 0
    last_6_balls_wickets: int = 0


# ─── Helper ──────────────────────────────────────────────────────────────────────
def _balls_info(overs: float):
    overs_completed = int(overs)
    balls_in_over = round((overs - overs_completed) * 10)
    total_bowled = overs_completed * 6 + balls_in_over
    balls_left = max(120 - total_bowled, 0)
    return overs_completed, balls_in_over, total_bowled, balls_left


def _win_prob_input(batting_team, bowling_team, city, target,
                    score, total_balls, balls_left, wickets_fallen):
    runs_left = max(target - score, 0)
    wickets_left = 10 - wickets_fallen
    crr = (score * 6) / total_balls if total_balls > 0 else 0
    rrr = (runs_left * 6) / max(balls_left, 1)
    return pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'target_runs': [target],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'crr': [crr],
        'rrr': [rrr]
    }), rrr


# ─── Routes ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "teams": teams,
        "cities": city_name
    })


@app.post("/predict")
async def predict_win(data: MatchData):
    if model is None:
        return {"error": "Model not loaded on the server."}

    overs_completed, balls_in_over, total_bowled, balls_left = _balls_info(data.overs)
    input_df, _ = _win_prob_input(
        data.batting_team, data.bowling_team, data.city,
        data.target, data.score, total_bowled, balls_left, data.wickets
    )

    try:
        result = model.predict_proba(input_df)[0]
        return {
            "batting_prob": round(result[1] * 100, 1),
            "bowling_prob": round(result[0] * 100, 1)
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/predict_ball", response_class=HTMLResponse)
async def serve_predict_ball_page(request: Request):
    return templates.TemplateResponse("predict_ball.html", {
        "request": request,
        "teams": teams,
        "cities": city_name
    })


@app.post("/predict_ball")
async def predict_ball(data: BallMatchData):
    if model2 is None:
        return {"error": "Ball-by-ball model not loaded on the server."}
    if model is None:
        return {"error": "Win probability model not loaded on the server."}

    # ── Feature engineering for current state ────────────────────────────────
    overs_completed, balls_in_over, total_bowled, balls_left = _balls_info(data.overs)
    _, rrr = _win_prob_input(
        data.batting_team, data.bowling_team, data.city,
        data.target, data.score, total_bowled, balls_left, data.wickets
    )

    wickets_fallen = data.wickets
    wickets_left = 10 - wickets_fallen

    is_powerplay = 1 if data.overs <= 6 else 0
    is_death    = 1 if data.overs > 15 else 0

    momentum_finisher = (
    is_death and
    data.last_18_balls_runs >= 30 and
    data.wickets >= 4
    ) 

    # ── STEP 1 : Ball model prediction ──────────────────────────────────────
    ball_input_df = pd.DataFrame({
        'batting_team':       [data.batting_team],
        'bowling_team':       [data.bowling_team],  
        'city':               [data.city],
        'target_runs':        [data.target],
        'current_score':      [data.score],
        'wickets':            [wickets_left],
        'is_powerplay':       [is_powerplay],
        'is_death':           [is_death],
        'momentum_finisher':  [momentum_finisher],
        'last_5_balls_runs':  [data.last_5_balls_runs],
        'last_18_balls_runs': [data.last_18_balls_runs],
        'dots_last_6':        [data.dots_last_6],
        'rrr':                [rrr],
        'partnership_runs':   [data.partnership_runs],
        'last_6_balls_wickets': [data.last_6_balls_wickets]
    })

    try:
        ball_probs = model2.predict_proba(ball_input_df)[0]
        classes    = model2.classes_          # e.g. [0, 1, 2, 3]
    except Exception as e:
        return {"error": f"Ball model prediction failed: {str(e)}"}

    # Probabilities for each class
    ball_prediction = {str(cls): round(float(prob) * 100, 2)
                       for cls, prob in zip(classes, ball_probs)}

    # Probabilistic sample (not argmax) for next event
    next_event = int(classes[np.argmax(ball_probs)])

    if next_event == 0:
        next_runs = 0

    elif next_event == 1:
        next_runs = int(np.random.choice([1, 2, 3]))   # 🔥 1–3 random

    elif next_event == 2:
        next_runs = int(np.random.choice([4, 6]))      # 🔥 4 or 6 random

    elif next_event == 3:
        next_runs = 0   # wicket

    # ── STEP 2 : Update match state ─────────────────────────────────────────
    new_score   = data.score + next_runs
    new_wickets = wickets_fallen + (1 if next_event == 3 else 0)

    new_balls_in_over = balls_in_over + 1
    new_overs_completed = overs_completed
    if new_balls_in_over == 6:
        new_overs_completed += 1
        new_balls_in_over = 0

    new_total_balls = new_overs_completed * 6 + new_balls_in_over
    new_balls_left  = max(120 - new_total_balls, 0)

    # ── STEP 3 : Win probability after next ball ────────────────────────────
    win_input, _ = _win_prob_input(
        data.batting_team, data.bowling_team, data.city,
        data.target, new_score, new_total_balls, new_balls_left, new_wickets
    )

    try:
        win_probs = model.predict_proba(win_input)[0]
    except Exception as e:
        return {"error": f"Win model prediction failed: {str(e)}"}

    return {
        "ball_prediction": ball_prediction,          # {"0": %, "1": %, "2": %, "3": %}
        "next_event":      str(next_event),
        "next_runs":       next_runs,
        "batting_prob":    round(float(win_probs[1]) * 100, 2),
        "bowling_prob":    round(float(win_probs[0]) * 100, 2)
    }