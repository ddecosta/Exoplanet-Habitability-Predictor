import pandas as pd
from physics import get_hz_boundaries, calculate_esi_score

FEATURE_COLUMNS = [
    'pl_orbper', 'st_teff', 'st_rad', 'sy_pnum', 'st_met', 'st_age',
    'st_logg', 'st_rotp', 'pl_orbincl', 'pl_orbeccen', 'pl_imppar',
    'st_mass', 'st_dens', 'st_vsin'
]

def load_and_prep_exoplanet_data(filepath='exoplanet_data.csv'):
    df = pd.read_csv(filepath)
    df['hz_inner'], df['hz_outer'] = get_hz_boundaries(df)
    df['habitability_zone_target'] = (
        (df['pl_orbsmax'] >= df['hz_inner']) &
        (df['pl_orbsmax'] <= df['hz_outer'])
    ).astype(int)
    df['esi_score'] = calculate_esi_score(df)
    
    return df.dropna(subset=['esi_score', 'habitability_zone_target']).copy()