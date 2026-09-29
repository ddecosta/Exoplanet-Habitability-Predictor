import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, precision_recall_curve, 
                             average_precision_score)
from xgboost import XGBClassifier

df = pd.read_csv('exoplanet_data.csv')

def get_hz_boundaries(data):
    t_star = data['st_teff'] - 5780
    i_c = [1.7763, 2.1301e-4, 2.524e-9, -5.287e-13, -6.248e-17]
    o_c = [0.3207, 4.2856e-5, 5.287e-10, -5.967e-13, -9.100e-17]

    def calc_seff(c):
        return c[0] + c[1]*t_star + c[2]*t_star**2 + c[3]*t_star**3 + c[4]*t_star**4

    seff_inner = calc_seff(i_c)
    seff_outer = calc_seff(o_c)

    inner_au = np.sqrt((10**data['st_lum']) / seff_inner)
    outer_au = np.sqrt((10**data['st_lum']) / seff_outer)
    return inner_au, outer_au

df['hz_inner'], df['hz_outer'] = get_hz_boundaries(df)
df['habitability_zone_target'] = (
    (df['pl_orbsmax'] >= df['hz_inner']) &
    (df['pl_orbsmax'] <= df['hz_outer'])
).astype(int)

features = [
    'pl_orbper', 'st_teff', 'st_rad', 'sy_pnum', 'st_met', 'st_age',
    'st_logg', 'st_rotp', 'pl_orbincl', 'pl_orbeccen', 'pl_imppar',
    'st_mass', 'st_dens', 'st_vsin'
]

X = df[features]
y = df['habitability_zone_target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

ratio = (len(y_train) - sum(y_train)) / sum(y_train)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost (GBDT)': XGBClassifier(n_estimators=100, scale_pos_weight=ratio, random_state=42, eval_metric='logloss'),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
plt.figure(figsize=(10, 8))

for name, clf in models.items():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', clf)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, y_prob)
    }

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)
    plt.plot(recall, precision, label=f'{name} (AP = {ap_score:.3f})')

plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title('Precision-Recall Curve: Targeting the Habitable Minority')
plt.legend()
plt.grid(True, alpha=0.3)

# Save figure BEFORE show() to avoid blank outputs
plt.savefig('habitability_precision_recall_score.png')
plt.close()

results_df = pd.DataFrame(results).T
print(results_df)
