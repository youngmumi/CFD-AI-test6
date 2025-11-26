# spline_xgb_hybrid.py
# aoa -> (cl, cd) 예측
# Spline(부드러운 baseline) + XGBoost(스파이크 residual) 하이브리드 모델

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from scipy.interpolate import UnivariateSpline
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
csv_path = "original.csv"      # <-- 네 CSV 파일 이름으로 바꿔줘 (aoa, cl, cd 컬럼)
outdir = "spline_xgb_output"
os.makedirs(outdir, exist_ok=True)

# -------------------------------------------------
# 2) 데이터 로드
# -------------------------------------------------
df = pd.read_csv(csv_path)

aoa_deg = df["aoa"].values.astype(float)
cl = df["cl"].values.astype(float)
cd = df["cd"].values.astype(float)

# -------------------------------------------------
# 3) 70% 학습 / 30% 검증 분할
#    (스플라인/잔차 모델 모두 train 데이터만 사용)
# -------------------------------------------------
aoa_train, aoa_val, cl_train, cl_val, cd_train, cd_val = train_test_split(
    aoa_deg,
    cl,
    cd,
    test_size=0.3,
    random_state=42,
    shuffle=True,
)

# -------------------------------------------------
# 4) Spline baseline 모델 학습 (cl, cd 각각)
#    UnivariateSpline(s>0)로 스파이크를 "부드럽게" 넘겨버림 → spike가 residual로 남게 됨
# -------------------------------------------------
# smoothing factor s는 데이터 스케일에 따라 조절 필요
# 대략 variance * (데이터 개수의 일부) 정도로 두면 괜찮은 편
var_cl = np.var(cl_train)
var_cd = np.var(cd_train)

s_cl = 0.2 * len(cl_train) * var_cl   # 필요시 숫자 조절 (0.1~1.0 배)
s_cd = 0.2 * len(cd_train) * var_cd

# aoa 기준으로 정렬 후 spline 학습 (UnivariateSpline은 x가 정렬되어 있는 게 안전)
order_train = np.argsort(aoa_train)
aoa_train_sorted = aoa_train[order_train]
cl_train_sorted = cl_train[order_train]
cd_train_sorted = cd_train[order_train]

spline_cl = UnivariateSpline(aoa_train_sorted, cl_train_sorted, s=s_cl)
spline_cd = UnivariateSpline(aoa_train_sorted, cd_train_sorted, s=s_cd)

# -------------------------------------------------
# 5) Train 데이터에서 residual 계산 (실제값 - spline 예측)
# -------------------------------------------------
cl_base_train = spline_cl(aoa_train)
cd_base_train = spline_cd(aoa_train)

res_cl_train = cl_train - cl_base_train
res_cd_train = cd_train - cd_base_train

# -------------------------------------------------
# 6) Residual을 예측하는 XGBoost / HistGB 학습
#    입력 특징은 간단히 aoa_deg 하나만 사용 (원하면 sin/cos 추가 가능)
# -------------------------------------------------
X_train_res = aoa_train.reshape(-1, 1)  # shape (N,1)

if XGB_AVAILABLE:
    xgb_params = dict(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
else:
    xgb_params = dict(
        max_depth=None,
        learning_rate=0.05,
        max_iter=400,
        random_state=42,
    )

xgb_res_cl = XGBRegressor(**xgb_params)
xgb_res_cd = XGBRegressor(**xgb_params)

xgb_res_cl.fit(X_train_res, res_cl_train)
xgb_res_cd.fit(X_train_res, res_cd_train)

# -------------------------------------------------
# 7) 검증 세트에서 하이브리드 예측
#    최종 예측 = spline(aoa) + XGB_residual(aoa)
# -------------------------------------------------
aoa_val_sorted_order = np.argsort(aoa_val)
aoa_val_sorted = aoa_val[aoa_val_sorted_order]

# baseline from spline
cl_base_val = spline_cl(aoa_val)
cd_base_val = spline_cd(aoa_val)

# residual prediction from XGBoost
X_val_res = aoa_val.reshape(-1, 1)
res_cl_val_pred = xgb_res_cl.predict(X_val_res)
res_cd_val_pred = xgb_res_cd.predict(X_val_res)

# 최종 예측
cl_val_pred = cl_base_val + res_cl_val_pred
cd_val_pred = cd_base_val + res_cd_val_pred

cl_true = cl_val
cd_true = cd_val

# -------------------------------------------------
# 8) 성능 지표 계산
# -------------------------------------------------
mse_cl = mean_squared_error(cl_true, cl_val_pred)
mse_cd = mean_squared_error(cd_true, cd_val_pred)
r2_cl = r2_score(cl_true, cl_val_pred)
r2_cd = r2_score(cd_true, cd_val_pred)

metrics = {
    "Model": "Spline + XGB residual hybrid",
    "Spline_s": {
        "s_cl": float(s_cl),
        "s_cd": float(s_cd),
    },
    "XGB_available": XGB_AVAILABLE,
    "CL": {"MSE": mse_cl, "R2": r2_cl},
    "CD": {"MSE": mse_cd, "R2": r2_cd},
}

print(json.dumps(metrics, indent=2))
with open(os.path.join(outdir, "metrics_spline_xgb.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# -------------------------------------------------
# 9) 그래프 저장 (aoa vs cl, aoa vs cd, true vs pred)
# -------------------------------------------------
# 정렬해서 그리기
order_val = np.argsort(aoa_val)
aoa_val_sorted = aoa_val[order_val]
cl_true_sorted = cl_true[order_val]
cl_pred_sorted = cl_val_pred[order_val]
cd_true_sorted = cd_true[order_val]
cd_pred_sorted = cd_val_pred[order_val]

# (1) aoa vs cl
plt.figure(figsize=(10, 4))
plt.plot(aoa_val_sorted, cl_true_sorted, label="true cl")
plt.plot(aoa_val_sorted, cl_pred_sorted, "--", label="hybrid pred cl")
plt.xlabel("aoa [deg]")
plt.ylabel("cl")
plt.title("Validation (Spline+XGB): aoa vs cl (true vs pred)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "aoa_vs_cl_val_spline_xgb.png"), dpi=150)
plt.close()

# (2) aoa vs cd
plt.figure(figsize=(10, 4))
plt.plot(aoa_val_sorted, cd_true_sorted, label="true cd")
plt.plot(aoa_val_sorted, cd_pred_sorted, "--", label="hybrid pred cd")
plt.xlabel("aoa [deg]")
plt.ylabel("cd")
plt.title("Validation (Spline+XGB): aoa vs cd (true vs pred)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "aoa_vs_cd_val_spline_xgb.png"), dpi=150)
plt.close()

# -------------------------------------------------
# 10) 모델 묶어서 저장 (나중에 예측용)
# -------------------------------------------------
bundle = {
    "spline_cl": spline_cl,
    "spline_cd": spline_cd,
    "xgb_res_cl": xgb_res_cl,
    "xgb_res_cd": xgb_res_cd,
}

model_path = os.path.join(outdir, "spline_xgb_model.joblib")
joblib.dump(bundle, model_path)

print(f"\nSaved hybrid model to: {model_path}")
print(f"Outputs saved in folder: {outdir}")
