import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('exoplanet_data.csv')

# Target Calculation
df['relative_flux'] = (10**df['st_lum']) / (df['pl_orbsmax']**2)

def calculate_esi_component(val, reference, weight):
    return (1 - np.abs((val - reference) / (val + reference)))**weight

df['esi_r'] = calculate_esi_component(df['pl_rade'], 1.0, 0.57)
df['esi_f'] = calculate_esi_component(df['relative_flux'], 1.0, 1.07)
df['esi_score'] = np.sqrt(df['esi_r'] * df['esi_f'])

df_clean = df.dropna(subset=['esi_score']).copy()

# Omit direct target inputs (pl_rade, pl_orbsmax, st_lum) to prevent leakage
features = [
    'pl_orbper', 'st_teff', 'st_rad', 'sy_pnum', 'st_met', 'st_age',
    'st_logg', 'st_rotp', 'pl_orbincl', 'pl_orbeccen', 'pl_imppar',
    'st_mass', 'st_dens', 'st_vsin'
]

X = df_clean[features]
y = df_clean['esi_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "SVR (RBF Kernel)": SVR(epsilon=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}
plt.figure(figsize=(15, 10))

for i, (name, model_obj) in enumerate(models.items(), 1):
    # Leakage-free preprocessing pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model_obj)
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    results[name] = r2
    
    plt.subplot(2, 4, i)
    plt.scatter(y_test, preds, alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f"{name}\nR^2: {r2:.3f} | MAE: {mae:.3f}")
    plt.xlabel('Actual')
    plt.ylabel('Pred')

plt.tight_layout()
plt.savefig('esi_regression_results.png')
plt.close()

print("\n--- Model Performance Summary (R^2 Score) ---")
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:20}: {score:.4f}")