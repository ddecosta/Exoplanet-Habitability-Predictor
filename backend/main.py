from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from processing import load_and_prep_exoplanet_data, FEATURE_COLUMNS

data_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup logic: Loads models and data, computes predictions, and caches
    the API response payload into memory once when the server boots up.
    """
    #Load data and calculate target properties
    df = load_and_prep_exoplanet_data('exoplanet_data.csv')
    
    try:
        #Load the pipeline payload
        pipeline_data = joblib.load('exoplanet_model_pipeline.joblib')
        
        clf_pipeline = pipeline_data['clf_pipeline']
        reg_pipeline = pipeline_data['reg_pipeline']
        features = pipeline_data.get('features', FEATURE_COLUMNS)

        #Ensure all required feature columns exist in the DataFrame
        for col in features:
            if col not in df.columns:
                df[col] = np.nan

        prob = clf_pipeline.predict_proba(df[features])[:, 1]
        esi = reg_pipeline.predict(df[features])
        
        df['potential'] = prob * esi
        df['hzProb'] = prob
        df['esi'] = esi

    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Pipeline loading failed ({e}). Defaulting to physics target fallback.")
        df['potential'] = 0.0
        df['hzProb'] = 0.0
        df['esi'] = df.get('esi_score', 0.0)

    #Merge with PHL Data for comparison
    try:
        phl_df = pd.read_csv('PHL_Data.csv')
        
        #Handle naming schema variations
        phl_col = 'P_NAME' if 'P_NAME' in phl_df.columns else 'Name'
        hab_col = 'P_HABITABLE' if 'P_HABITABLE' in phl_df.columns else 'Habitable'
        
        phl_df['pl_name_clean'] = phl_df[phl_col].astype(str).str.strip().str.lower()
        df['pl_name_clean'] = df['pl_name'].astype(str).str.strip().str.lower()
        
        combined = pd.merge(
            df, 
            phl_df[['pl_name_clean', hab_col]], 
            on='pl_name_clean', 
            how='left'
        )
        combined['phl_status'] = combined[hab_col]
    except FileNotFoundError:
        combined = df.copy()
        combined['phl_status'] = np.nan

    #Pre-format top planets payload for GET /api/top-planets
    top_planets = combined.sort_values(by='potential', ascending=False).head(15)
    
    top_planets_output = []
    for _, row in top_planets.iterrows():
        top_planets_output.append({
            "name": row['pl_name'],
            "potential": float(row.get('potential', 0.0)),
            "hzProb": float(row.get('hzProb', 0.0)),
            "esi": float(row.get('esi', 0.0)),
            "phl_status": str(row['phl_status']) if pd.notnull(row.get('phl_status')) else "Not Listed"
        })
        
    data_cache['top_planets'] = top_planets_output

    #Pre-format EDA timeline & scatter payload for GET /api/eda-data
    timeline = df['disc_year'].value_counts().sort_index().reset_index()
    timeline.columns = ['year', 'count']
    
    scatter = df.sample(n=min(300, len(df)))[['st_teff', 'pl_rade', 'pl_name']].dropna()
    
    data_cache['eda'] = {
        "timeline": timeline.to_dict(orient="records"),
        "scatter": scatter.to_dict(orient="records")
    }

    yield  #Server runs while execution remains suspended here
    
    data_cache.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/top-planets")
async def get_planets():
    """Returns top candidate exoplanets instantly from in-memory cache."""
    return data_cache.get('top_planets', [])


@app.get("/api/eda-data")
async def get_eda_data():
    """Returns EDA chart data instantly from in-memory cache."""
    return data_cache.get('eda', {"timeline": [], "scatter": []})