# Exoplanet Habitability Predictor

A machine learning application that predicts and prioritizes exoplanet habitability candidates from the NASA Exoplanet Archive using a Two-Hurdle model architecture (XGBoost Classifier + Random Forest Regressor) and visualizes results in an interactive web dashboard.


## Project Overview

With over 6,000 confirmed exoplanets discovered, analyzing each candidate manually is resource-intensive due to incomplete datasets and extreme class imbalance (~1% habitable candidates). 

This project tackles class imbalance by implementing a **Two-Hurdle Machine Learning Model** based on John Cragg's statistical approach:
1. **Hurdle 1 (Filtering):** An XGBoost Classifier determines whether an exoplanet resides within its star's Habitable Zone (HZ) using Kopparapu et al. boundary equations.
2. **Hurdle 2 (Scoring):** A Random Forest Regressor predicts the Earth Similarity Index (ESI) score based on physical characteristics (radius, bulk density, escape velocity, surface temperature).
3. **Final Prioritization:** Outputs an Overall Habitability Score (HZ Probability * Predicted ESI) to deliver a prioritized ranking for astronomical target follow-ups.

<img width="603" height="345" alt="image" src="https://github.com/user-attachments/assets/ba2fb368-8ccb-4c41-b5b5-06176ce877b4" />


## Model Performance & Metrics

The model was validated against the **Habitable Worlds Catalog (HWC)** from the Planetary Habitability Laboratory (PHL):

- **Precision@5:** **80.00%** (4/5 top candidates matched)
- **Precision@10:** **60.00%**
- **Precision@20:** **50.00%**

## Further Reading

For a more in-depth discussion of the evaluation methods used to determine this optimal model architecture, an explanation of the problems encountered (such as severe class imbalance and theoretical metric limitations), and a full list of scientific references used, please see the attached [Project Report](Project%20Report.pdf).
