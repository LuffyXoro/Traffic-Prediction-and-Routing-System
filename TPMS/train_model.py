import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import pickle
import joblib
import os

data = pd.read_csv("TPMS/data/final_traffic_dataset_with_location_features.csv")
target_col = "Congestion_Level"


le_target = LabelEncoder()
data[target_col] = le_target.fit_transform(data[target_col])

data = data.sort_values(by=["City", "Road_Name", "hour"])

# feature's

data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

data["prev_congestion"] = data.groupby(["City", "Road_Name"])[target_col].shift(1).fillna(0)
data["rolling_congestion"] = (
    data.groupby(["City", "Road_Name"])[target_col]
    .transform(lambda x: x.rolling(3).mean())
).fillna(0)

data["traffic_speed_ratio"] = data["Traffic_Density"] / (data["Avg_Speed"] + 1)
data["speed_efficiency"]    = data["Avg_Speed"] / (data["Traffic_Density"] + 1)
data["density_x_rush"]      = data["Traffic_Density"] * data["Rush_Hour"]
data["peak_hour_flag"]      = data["hour"].apply(lambda x: 1 if (7 <= x <= 10 or 17 <= x <= 20) else 0)

data = data.drop(columns=["Timestamp", "Area_Name", "Latitude", "Longitude"], errors='ignore')

os.makedirs("TPMS/models", exist_ok=True)

label_encoders = {}
for col in data.columns:
    if data[col].dtype == "object" and col != target_col:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

joblib.dump(label_encoders, "TPMS/models/label_encoders_all.pkl")


feature_columns = data.drop(columns=[target_col]).columns.tolist()
joblib.dump(feature_columns, "TPMS/models/model_columns.pkl")

data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data)
joblib.dump(scaler, "TPMS/models/scaler.pkl")


X, y = [], []
seq = 30

for _, group in data.groupby(["City", "Road_Name"]):
    group = group.sort_values("hour")
    group = group.drop(columns=["Road_Name"], errors='ignore')
    X_group = group.drop(columns=[target_col])
    y_group = group[target_col]
    for i in range(len(X_group) - seq):
        X.append(X_group.iloc[i:i+seq].values)
        y.append(y_group.iloc[i+seq])

X = np.array(X)
y = np.array(y)

seq_feature_cols = data.drop(columns=["Road_Name", target_col], errors='ignore').columns.tolist()
joblib.dump(seq_feature_cols, "TPMS/models/seq_feature_cols.pkl")
print(f"LSTM input columns ({len(seq_feature_cols)}): {seq_feature_cols}")

# train-test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# lstm

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
model.fit(X_train, y_train, epochs=40, batch_size=32,
          validation_data=(X_test, y_test), callbacks=[early_stop])


y_pred_classes = np.argmax(model.predict(X_test), axis=1).astype(int)
y_test_int     = y_test.astype(int)

accuracy  = accuracy_score(y_test_int, y_pred_classes)
precision = precision_score(y_test_int, y_pred_classes, average='weighted', zero_division=0)
recall    = recall_score(y_test_int, y_pred_classes, average='weighted')
f1        = f1_score(y_test_int, y_pred_classes, average='weighted')
print(f"\n🔹 LSTM  Acc:{accuracy:.4f}  Prec:{precision:.4f}  Rec:{recall:.4f}  F1:{f1:.4f}")

#save 

model.save("TPMS/models/traffic_congestion_model.keras")

with open("TPMS/models/metrics.txt", "w") as f:
    f.write(f"Accuracy:{accuracy}\nPrecision:{precision}\nRecall:{recall}\nF1:{f1}\n")

with open("TPMS/models/label_encoder.pkl", "wb") as f:
    pickle.dump(le_target, f)

# baseline

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_flat, y_test_int[:len(X_train_flat)] if len(X_train_flat) != len(y_train) else y_train.astype(int))
lr_model.fit(X_train_flat, y_train.astype(int))
lr_preds = lr_model.predict(X_test_flat)

lr_acc  = accuracy_score(y_test_int, lr_preds)
lr_prec = precision_score(y_test_int, lr_preds, average='weighted')
lr_rec  = recall_score(y_test_int, lr_preds, average='weighted')
lr_f1   = f1_score(y_test_int, lr_preds, average='weighted')
print(f"\n🔹 LR    Acc:{lr_acc:.4f}  Prec:{lr_prec:.4f}  Rec:{lr_rec:.4f}  F1:{lr_f1:.4f}")

with open("model_comparison_results.txt", "w") as f:
    f.write("MODEL PERFORMANCE COMPARISON\n" + "="*40 + "\n\n")
    f.write(f"Logistic Regression\nAccuracy:{lr_acc:.4f}\nPrecision:{lr_prec:.4f}\nRecall:{lr_rec:.4f}\nF1:{lr_f1:.4f}\n\n")
    f.write(f"LSTM\nAccuracy:{accuracy:.4f}\nPrecision:{precision:.4f}\nRecall:{recall:.4f}\nF1:{f1:.4f}\n\n")
    conclusion = "LSTM outperforms Logistic Regression." if accuracy > lr_acc else "Logistic Regression performs better."
    f.write("="*40 + f"\nConclusion: {conclusion}\n")

print("\nAll artifacts saved to TPMS/models/")