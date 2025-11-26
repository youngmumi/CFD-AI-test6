# interpolate_and_train.py
# 1) 기존 360개 데이터를 3600개로 보간
# 2) 보간된 데이터로 Spline + XGBoost 하이브리드 학습
# 3) 모델 및 그래프 저장

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import CubicSpline, UnivariateSpline
import joblib

# XGBoost import
XGB_AVAILABLE = True
try:
    from xgboost import XGBRegressor
except Exception:
    XGB_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor

# -------------------------------------------------
# 1) 경로
# -------------------------------------------------
csv_path = "original.csv"     # 네 원본 CSV 파일
outdir = "interp_3600_output"
os.makedirs(outdir, exist_ok=True)

# -------------------------------------------------
# 2) 원본 데이터 로드
# -------------------------------------------------
df = pd.read_csv(csv_path)

aoa = df["aoa"].values.astype(float)
cl = df["cl"].values.astype(float)
cd = df["cd"].values.astype(float)

# aoa는 반드시 오름차순 정렬해야 spline 안정
order = np.argsort(aoa)
aoa = aoa[order]
cl = cl[order]
cd = cd[order]

# -------------------------------------------------
# 3) CubicSpline 보간으로 데이터 360 → 3600개 확장
# -------------------------------------------------
N_new = 3600
aoa_new = np.linspace(aoa.min(), aoa.max(), N_new)

cs_cl = CubicSpline(aoa, cl)
cs_cd = CubicSpline(aoa, cd)

cl_new = cs_cl(aoa_new)
cd_new = cs_cd(aoa_new)

# 보간된 데이터 저장
interp_df = pd.DataFrame({
    "aoa": aoa_new,
    "cl": cl_new,
    "cd": cd_new,
})
interp_df.to_csv(os.path.join(outdir, "interp_3600.csv"), index=False)

print(f"Interpolated dataset saved: {outdir}/interp_3600.csv")

# -------------------------------------------------
# 4) 3600개 데이터로 train/val 분할
# -------------------------------------------------
aoa_train, aoa_val, cl_train, cl_val, cd_train, cd_val = train_test_split(
    aoa_new,
    cl_new,
    cd_new,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

# -------------------------------------------------
# 5) smoothing spline (baseline)
# -------------------------------------------------
var_cl = np.var(cl_train)
var_cd = np.var(cd_train)

s_cl = 0.1 * len(cl_train) * var_cl      # 필요하면 0.05 ~ 0.5 조절
s_cd = 0.1 * len(cd_train) * var_cd

order_train = np.argsort(aoa_train)

spline_cl = UnivariateSpline(aoa_train[order_train], cl_train[order_train], s=s_cl)
spline_cd = UnivariateSpline(aoa_train[order_train], cd_train[order_train], s=s_cd)

# spline baseline 예측
cl_base_train = spline_cl(aoa_train)
cd_base_train = spline_cd(aoa_train)

# -------------------------------------------------
# 6) Residual = 실제 - baseline
# -------------------------------------------------
res_cl_train = cl_train - cl_base_train
res_cd_train = cd_train - cd_base_train

X_train = aoa_train.reshape(-1, 1)

# -------------------------------------------------
# 7) XGBoost residual 모델 학습
# -------------------------------------------------
if XGB_AVAILABLE:
    xgb_params = dict(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
else:
    xgb_params = dict(
        max_depth=None,
        learning_rate=0.05,
        max_iter=600,
        random_state=42,
    )

xgb_res_cl = XGBRegressor(**xgb_params)
xgb_res_cd = XGBRegressor(**xgb_params)

xgb_res_cl.fit(X_train, res_cl_train)
xgb_res_cd.fit(X_train, res_cd_train)

# -------------------------------------------------
# 8) Validation: Hybrid = spline + residual
# -------------------------------------------------
X_val = aoa_val.reshape(-1, 1)

cl_base_val = spline_cl(aoa_val)
cd_base_val = spline_cd(aoa_val)

res_cl_val_pred = xgb_res_cl.predict(X_val)
res_cd_val_pred = xgb_res_cd.predict(X_val)

cl_val_pred = cl_base_val + res_cl_val_pred
cd_val_pred = cd_base_val + res_cd_val_pred

# -------------------------------------------------
# 9) 성능 평가
# -------------------------------------------------
mse_cl = mean_squared_error(cl_val, cl_val_pred)
mse_cd = mean_squared_error(cd_val, cd_val_pred)
r2_cl = r2_score(cl_val, cl_val_pred)
r2_cd = r2_score(cd_val, cd_val_pred)

metrics = {
    "Model": "CubicSpline + XGB Hybrid (with 3600 interpolated data)",
    "CL": {"MSE": mse_cl, "R2": r2_cl},
    "CD": {"MSE": mse_cd, "R2": r2_cd},
    "smoothing": {"s_cl": float(s_cl), "s_cd": float(s_cd)},
    "XGB_available": XGB_AVAILABLE,
}

with open(os.path.join(outdir, "metrics_interp_hybrid.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(json.dumps(metrics, indent=2))

# -------------------------------------------------
# 10) 그래프 저장
# -------------------------------------------------
order_val = np.argsort(aoa_val)

# CL
plt.figure(figsize=(10,4))
plt.plot(aoa_val[order_val], cl_val[order_val], label="true cl")
plt.plot(aoa_val[order_val], cl_val_pred[order_val], "--", label="hybrid cl")
plt.legend()
plt.title("CL Validation (Hybrid)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(outdir, "cl_val_hybrid.png"), dpi=150)
plt.close()

# CD
plt.figure(figsize=(10,4))
plt.plot(aoa_val[order_val], cd_val[order_val], label="true cd")
plt.plot(aoa_val[order_val], cd_val_pred[order_val], "--", label="hybrid cd")
plt.legend()
plt.title("CD Validation (Hybrid)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(outdir, "cd_val_hybrid.png"), dpi=150)
plt.close()

# -------------------------------------------------
# 11) 전체 모델 저장 (spline + XGB)
# -------------------------------------------------
bundle = {
    "spline_cl": spline_cl,
    "spline_cd": spline_cd,
    "xgb_res_cl": xgb_res_cl,
    "xgb_res_cd": xgb_res_cd,
}

joblib.dump(bundle, os.path.join(outdir, "spline_xgb_model_3600.joblib"))

print(f"Saved trained hybrid model to {outdir}/spline_xgb_model_3600.joblib")
