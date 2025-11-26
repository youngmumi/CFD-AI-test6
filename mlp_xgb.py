# mlp_xgb_ensemble.py
# aoa -> (cl, cd) 예측
# 70% train / 30% val
# MLP(기본) + XGBoost(잔차 보정) 앙상블

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# XGBoost가 없으면 HistGradientBoostingRegressor로 대체
XGB_AVAILABLE = True
try:
    from xgboost import XGBRegressor
except Exception:
    XGB_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor

# -------------------------------------------------
# 1) 경로 설정
# -------------------------------------------------
csv_path = "original.csv"     # <-- 네 CSV 파일 이름으로 바꿔줘 (aoa, cl, cd 컬럼)
outdir = "ensemble_output"
os.makedirs(outdir, exist_ok=True)

# -------------------------------------------------
# 2) 데이터 로드
# -------------------------------------------------
df = pd.read_csv(csv_path)

aoa_deg = df["aoa"].values.astype(float)
cl = df["cl"].values.astype(float)
cd = df["cd"].values.astype(float)

# -------------------------------------------------
# 3) 특징 생성: aoa -> sin, cos (주기성 반영)
# -------------------------------------------------
aoa_rad = np.deg2rad(aoa_deg)
X_raw = np.column_stack([np.sin(aoa_rad), np.cos(aoa_rad)])  # 2차원 입력
Y_raw = np.column_stack([cl, cd])                           # 2차원 출력

# -------------------------------------------------
# 4) 70% 학습 / 30% 검증 분할
# -------------------------------------------------
X_train_raw, X_val_raw, Y_train_raw, Y_val_raw, aoa_train, aoa_val = train_test_split(
    X_raw,
    Y_raw,
    aoa_deg,
    test_size=0.3,
    random_state=42,
    shuffle=True,
)

# -------------------------------------------------
# 5) 스케일링 (MLP용)
# -------------------------------------------------
X_scaler = StandardScaler()
Y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train_raw)
X_val = X_scaler.transform(X_val_raw)

Y_train = Y_scaler.fit_transform(Y_train_raw)
Y_val = Y_scaler.transform(Y_val_raw)

# -------------------------------------------------
# 6) MLP 기본 모델 학습 (cl, cd 동시 예측)
# -------------------------------------------------
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    max_iter=5000,
    early_stopping=True,
    n_iter_no_change=30,
    random_state=42,
)

mlp.fit(X_train, Y_train)

# 학습 데이터에서 MLP 기본 예측 (잔차 학습용)
Y_base_train_scaled = mlp.predict(X_train)
Y_base_train = Y_scaler.inverse_transform(Y_base_train_scaled)

# 잔차 = 실제값 - 기본예측
residual_train = Y_train_raw - Y_base_train
res_cl_train = residual_train[:, 0]
res_cd_train = residual_train[:, 1]

# -------------------------------------------------
# 7) 잔차를 예측하는 XGBoost / HistGB 모델 학습
#    (입력은 sin, cos 그대로 사용)
# -------------------------------------------------
if XGB_AVAILABLE:
    xgb_params = dict(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
else:
    # HistGradientBoostingRegressor 기본값도 충분히 강력함
    xgb_params = dict(
        max_depth=None,
        learning_rate=0.05,
        max_iter=400,
        random_state=42,
    )

xgb_res_cl = XGBRegressor(**xgb_params)
xgb_res_cd = XGBRegressor(**xgb_params)

xgb_res_cl.fit(X_train_raw, res_cl_train)
xgb_res_cd.fit(X_train_raw, res_cd_train)

# -------------------------------------------------
# 8) 검증 세트에서 앙상블 예측
#    최종 예측 = MLP 기본예측 + XGBoost 잔차예측
# -------------------------------------------------
# (1) MLP 기본 예측
Y_base_val_scaled = mlp.predict(X_val)
Y_base_val = Y_scaler.inverse_transform(Y_base_val_scaled)

# (2) 잔차 예측
res_cl_val_pred = xgb_res_cl.predict(X_val_raw)
res_cd_val_pred = xgb_res_cd.predict(X_val_raw)

residual_val_pred = np.column_stack([res_cl_val_pred, res_cd_val_pred])

# (3) 최종 앙상블 예측
Y_val_pred = Y_base_val + residual_val_pred

cl_true = Y_val_raw[:, 0]
cd_true = Y_val_raw[:, 1]
cl_pred = Y_val_pred[:, 0]
cd_pred = Y_val_pred[:, 1]

# -------------------------------------------------
# 9) 성능 지표 계산
# -------------------------------------------------
mse_cl = mean_squared_error(cl_true, cl_pred)
mse_cd = mean_squared_error(cd_true, cd_pred)
r2_cl = r2_score(cl_true, cl_pred)
r2_cd = r2_score(cd_true, cd_pred)

metrics = {
    "Model": "MLP + XGB residual ensemble",
    "MLP_params": {
        "hidden_layer_sizes": (64, 64),
        "activation": "relu",
        "alpha": 1e-4,
    },
    "XGB_available": XGB_AVAILABLE,
    "CL": {"MSE": mse_cl, "R2": r2_cl},
    "CD": {"MSE": mse_cd, "R2": r2_cd},
}

print(json.dumps(metrics, indent=2))

with open(os.path.join(outdir, "metrics_ensemble.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# -------------------------------------------------
# 10) 그래프 저장 (aoa vs cl, aoa vs cd)
# -------------------------------------------------
order = np.argsort(aoa_val)
aoa_sorted = aoa_val[order]
cl_true_sorted = cl_true[order]
cl_pred_sorted = cl_pred[order]
cd_true_sorted = cd_true[order]
cd_pred_sorted = cd_pred[order]

# aoa vs cl
plt.figure(figsize=(10, 4))
plt.plot(aoa_sorted, cl_true_sorted, label="true cl")
plt.plot(aoa_sorted, cl_pred_sorted, "--", label="ensemble pred cl")
plt.xlabel("aoa [deg]")
plt.ylabel("cl")
plt.title("Validation (ensemble): aoa vs cl (true vs pred)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "aoa_vs_cl_val_ensemble.png"), dpi=150)
plt.close()

# aoa vs cd
plt.figure(figsize=(10, 4))
plt.plot(aoa_sorted, cd_true_sorted, label="true cd")
plt.plot(aoa_sorted, cd_pred_sorted, "--", label="ensemble pred cd")
plt.xlabel("aoa [deg]")
plt.ylabel("cd")
plt.title("Validation (ensemble): aoa vs cd (true vs pred)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "aoa_vs_cd_val_ensemble.png"), dpi=150)
plt.close()

# -------------------------------------------------
# 11) 모델 + 스케일러 묶어서 저장
# -------------------------------------------------
bundle = {
    "mlp": mlp,
    "xgb_res_cl": xgb_res_cl,
    "xgb_res_cd": xgb_res_cd,
    "X_scaler": X_scaler,
    "Y_scaler": Y_scaler,
}

model_path = os.path.join(outdir, "ensemble_model.joblib")
joblib.dump(bundle, model_path)

print(f"\nSaved ensemble model to: {model_path}")
print(f"Outputs saved in folder: {outdir}")
