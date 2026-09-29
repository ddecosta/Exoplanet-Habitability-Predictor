import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ndcg_score

from processing import load_and_prep_exoplanet_data, FEATURE_COLUMNS

df_unified = load_and_prep_exoplanet_data('exoplanet_data.csv')

X = df_unified[FEATURE_COLUMNS]
y_hz = df_unified['habitability_zone_target']
y_esi = df_unified['esi_score']


#Stratified split to preserve class ratio
X_train, X_test, y_hz_train, y_hz_test, y_esi_train, y_esi_test = train_test_split(
    X, y_hz, y_esi, test_size=0.2, random_state=42, stratify=y_hz
)

#Calculate ratio for XGBoost scale_pos_weight
pos_ratio = (len(y_hz_train) - sum(y_hz_train)) / sum(y_hz_train)

#Construct pipelines
clf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(
        n_estimators=100, 
        scale_pos_weight=pos_ratio, 
        random_state=42, 
        eval_metric='logloss'
    ))
])

reg_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

#Train models
clf_pipeline.fit(X_train, y_hz_train)
reg_pipeline.fit(X_train, y_esi_train)

#Predict probabilities and regression values on test set
hz_probability = clf_pipeline.predict_proba(X_test)[:, 1]
predicted_esi = reg_pipeline.predict(X_test)

#Calculate composite habitability score
robust_score = hz_probability * predicted_esi


#Report
report = pd.DataFrame(index=X_test.index)
report['pl_name'] = df_unified.loc[X_test.index, 'pl_name']
report['pl_name_clean'] = report['pl_name'].str.strip().str.lower()
report['HZ_Prob'] = hz_probability
report['Pred_ESI'] = predicted_esi
report['Real_ESI'] = y_esi_test.values
report['Robust_Score'] = robust_score

report_sorted = report.sort_values(by='Robust_Score', ascending=False)

print("\n--- Top 10 Predicted Habitable Candidates ---")
print(report_sorted[['pl_name', 'Robust_Score', 'HZ_Prob', 'Pred_ESI', 'Real_ESI']].head(10).to_string(index=False))

#Feature Importances from RandomForest Regressor
rf_model = reg_pipeline.named_steps['regressor']
feature_importance_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS, 
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importances (Random Forest ESI) ---")
print(feature_importance_df.to_string(index=False))

#Export model pipelines
pipeline_payload = {
    'features': FEATURE_COLUMNS,
    'clf_pipeline': clf_pipeline,
    'reg_pipeline': reg_pipeline
}

joblib.dump(pipeline_payload, 'exoplanet_model_pipeline.joblib')
print("\n Successfully saved trained pipeline payload to 'exoplanet_model_pipeline.joblib'")

#Evaluate against Planetary Habitatbility Laboratory (PHL) dataset
df_phl = pd.read_csv('PHL_Data_Habitable.csv')
phl_habitable_names = set(df_phl['Name'].str.strip().str.lower().unique())


def evaluate_strictly(ranked_df, expert_set, k=10):
    """Evaluates top K predictions against PHL consensus matches."""
    top_k_df = ranked_df.head(k)
    top_k_cleans = set(top_k_df['pl_name_clean'])
    hits = top_k_cleans.intersection(expert_set)
    
    precision = len(hits) / k
    proper_matches = top_k_df[top_k_df['pl_name_clean'].isin(hits)]['pl_name'].unique().tolist()
    return precision, proper_matches


print("\n--- Model vs. PHL Consensus Evaluation ---")
for k in [5, 10, 20]:
    precision, matches = evaluate_strictly(report_sorted, phl_habitable_names, k=k)
    print(f"Precision@{k:2}: {precision:.2%}")
    if matches:
        print(f"   Validated Matches: {', '.join(sorted(matches))}")
    else:
        print("   Validated Matches: None found")

#Highly ranked by ML but missing from PHL catalog
top_20_cleans = set(report_sorted['pl_name_clean'].head(20))
hot_takes_cleans = top_20_cleans - phl_habitable_names
hot_takes_proper = report_sorted[report_sorted['pl_name_clean'].isin(hot_takes_cleans)]['pl_name'].head(5).tolist()

print("\n--- Model 'Hot Takes' (High ML Rank, absent in PHL) ---")
print(hot_takes_proper)


#NDCG Ranking
def calculate_ndcg(report_df, expert_set, k=20):
    relevance = report_df['pl_name_clean'].apply(lambda x: 1 if x in expert_set else 0).values
    scores = report_df['Robust_Score'].values
    
    return ndcg_score([relevance], [scores], k=k)


ndcg_val = calculate_ndcg(report_sorted, phl_habitable_names, k=20)
print("\n--- Ranking Performance ---")
print(f"NDCG@20: {ndcg_val:.4f}")