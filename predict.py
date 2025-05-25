import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib

# Expanded hardness dataset
data_hardness = {
    'Material': ['EN-8'] * 8 + ['Mild Steel'] * 8,
    'Current': [120, 125, 130, 135, 140, 145, 150, 155, 120, 125, 130, 135, 140, 145, 150, 155],
    'Heat_Input': [0.8, 0.816, 0.832, 0.848, 0.864, 0.888, 0.912, 0.912, 0.8, 0.816, 0.832, 0.848, 0.864, 0.888, 0.912, 0.912],
    'Carbon': [0.37] * 8 + [0.23] * 8,
    'Manganese': [0.8] * 8 + [1.0] * 8,
    'Hardness': [331, 339.5, 348, 356.5, 365, 366.5, 368, 369.5, 310, 320.75, 331.5, 342.25, 353, 355, 357, 359]
}
df_hardness = pd.DataFrame(data_hardness)

# Expanded oxidation rate dataset
data_oxidation = {
    'Material': (['EN-8'] * 5 + ['Mild Steel'] * 5) * 3,
    'Current': [120, 125, 130, 140, 160, 120, 125, 130, 140, 160] * 3,
    'Heat_Input': [0.8, 0.816, 0.832, 0.864, 0.912, 0.8, 0.816, 0.832, 0.864, 0.912] * 3,
    'Soaking_Time': [5] * 10 + [10] * 10 + [15] * 10,
    'Carbon': ([0.37] * 5 + [0.23] * 5) * 3,
    'Manganese': ([0.8] * 5 + [1.0] * 5) * 3,
    'Oxidation_Rate': [
        0.002603, 0.002526, 0.0024495, 0.002296, 0.002163, 0.002692, 0.002636, 0.002580, 0.002580, 0.002237,
        0.005206, 0.005051, 0.004896, 0.004592, 0.004326, 0.005384, 0.005270, 0.005156, 0.005156, 0.004474,
        0.008076, 0.007830, 0.007584, 0.007584, 0.006711, 0.008232, 0.008021, 0.007809, 0.007809, 0.006489
    ]
}
df_oxidation = pd.DataFrame(data_oxidation)

# Encode categorical variable (Material)
df_hardness['Material'] = df_hardness['Material'].map({'EN-8': 0, 'Mild Steel': 1})
df_oxidation['Material'] = df_oxidation['Material'].map({'EN-8': 0, 'Mild Steel': 1})

# Features and target for hardness
X_hardness = df_hardness[['Material', 'Current', 'Heat_Input', 'Carbon', 'Manganese']]
y_hardness = df_hardness['Hardness']

# Features and target for oxidation
X_oxidation = df_oxidation[['Material', 'Current', 'Heat_Input', 'Soaking_Time', 'Carbon', 'Manganese']]
y_oxidation = df_oxidation['Oxidation_Rate']

# Split data into training and testing sets
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hardness, y_hardness, test_size=0.2, random_state=42)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_oxidation, y_oxidation, test_size=0.2, random_state=42)

# Train and save Linear Regression model for hardness
lr_hardness = LinearRegression()
lr_hardness.fit(X_train_h, y_train_h)
joblib.dump(lr_hardness, 'lr_hardness_model.pkl')

# Train and save Random Forest model for oxidation
rf_oxidation = RandomForestRegressor(random_state=42)
rf_oxidation.fit(X_train_o, y_train_o)
joblib.dump(rf_oxidation, 'rf_oxidation_model.pkl')

print("Models trained and saved successfully.")
