# mini_mlp_tuned.py
# 70% train / 30% val, 간단 튜닝 + 스케일링 + 그래프 + 모델 저장

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

# -------------------------------------------------
# 1) 경로 설정
# -------------------------------------------------
csv_path = "original.csv"     # <-- 너 CSV 이름으로 바꿔줘
outdir = "mini_mlp_output"
os.makedirs(outdir, exist_ok=True)

# -------------------------------------------------
# 2) 데이터 로드 (aoa, cl, cd 컬럼 가정)
# -------------------------------------------------
df = pd.read_csv(csv_path)

aoa_deg = df["aoa"].values.astype(float)
cl = df["cl"].values.astype(float)
cd = df["cd"].values.astype(float)

# -------------------------------------------------
# 3) 특징 생성: aoa(deg) -> sin, cos
# -------------------------------------------------
aoa_rad = np.deg2rad(aoa_deg)
X_raw = np.column_stack([np.sin(aoa_rad), np.cos(aoa_rad)])
Y_raw = np.column_stack([cl, cd])

# -------------------------------------------------
# 4) 70% 학습 / 30% 검증으로 분할
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
# 5) 입력/출력 스케일링
#    - 신경망 성능, 수렴 속도 ↑
# -------------------------------------------------
X_scaler = StandardScaler()
Y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train_raw)
X_val = X_scaler.transform(X_val_raw)

Y_train = Y_scaler.fit_transform(Y_train_raw)
Y_val = Y_scaler.transform(Y_val_raw)

# -------------------------------------------------
# 6) 간단한 하이퍼파라미터 튜닝
#    여러 조합 중 검증 R² 평균(best_R2)을 최대화하는 모델 선택
# -------------------------------------------------
candidates = [
    {"hidden_layer_sizes": (32, 32), "alpha": 1e-3, "activation": "tanh"},
    {"hidden_layer_sizes": (64, 64), "alpha": 1e-3, "activation": "tanh"},
    {"hidden_layer_sizes": (64, 64), "alpha": 1e-4, "activation": "relu"},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "activation": "relu"},
]

best_model = None
best_score = -np.inf
best_params = None

for i, params in enumerate(candidates, 1):
    print(f"\n=== Candidate {i}: {params} ===")
    model = MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        solver="adam",
        alpha=params["alpha"],
        max_iter=5000,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42,
    )

    model.fit(X_train, Y_train)

    # 검증 세트 예측 (스케일 되있는 값 → 역변환)
    Y_val_pred_scaled = model.predict(X_val)
    Y_val_pred = Y_scaler.inverse_transform(Y_val_pred_scaled)

    cl_true = Y_val_raw[:, 0]
    cd_true = Y_val_raw[:, 1]
    cl_pred = Y_val_pred[:, 0]
    cd_pred = Y_val_pred[:, 1]

    r2_cl = r2_score(cl_true, cl_pred)
    r2_cd = r2_score(cd_true, cd_pred)
    score_mean = (r2_cl + r2_cd) / 2.0

    print(f"R2_cl = {r2_cl:.6f}, R2_cd = {r2_cd:.6f}, mean = {score_mean:.6f}")

    if score_mean > best_score:
        best_score = score_mean
        best_model = model
        best_params = {
            "hidden_layer_sizes": params["hidden_layer_sizes"],
            "alpha": params["alpha"],
            "activation": params["activation"],
            "R2_cl": r2_cl,
            "R2_cd": r2_cd,
            "R2_mean": score_mean,
        }

print("\n=== Best Params ===")
print(json.dumps(best_params, indent=2))

# -------------------------------------------------
# 7) 베스트 모델로 최종 평가 및 그래프 저장
# -------------------------------------------------
Y_val_pred_scaled = best_model.predict(X_val)
Y_val_pred = Y_scaler.inverse_transform(Y_val_pred_scaled)

cl_true = Y_val_raw[:, 0]
cd_true = Y_val_raw[:, 1]
cl_pred = Y_val_pred[:, 0]
cd_pred = Y_val_pred[:, 1]

mse_cl = mean_squared_error(cl_true, cl_pred)
mse_cd = mean_squared_error(cd_true, cd_pred)
r2_cl = r2_score(cl_true, cl_pred)
r2_cd = r2_score(cd_true, cd_pred)

metrics = {
    "Best Params": best_params,
    "Final Metrics": {
        "CL": {"MSE": mse_cl, "R2": r2_cl},
        "CD": {"MSE": mse_cd, "R2": r2_cd},
    },
}

print("\n=== Final Metrics ===")
print(json.dumps(metrics, indent=2))

with open(os.path.join(outdir, "metrics_tuned.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# aoa 기준 정렬해서 그래프 그리기
order = np.argsort(aoa_val)
aoa_sorted = aoa_val[order]
cl_true_sorted = cl_true[order]
cl_pred_sorted = cl_pred[order]
cd_true_sorted = cd_true[order]
cd_pred_sorted = cd_pred[order]

# (1) aoa vs cl
plt.figure(figsize=(10, 4))
plt.plot(aoa_sorted, cl_true_sorted, label="true cl")
plt.plot(aoa_sorted, cl_pred_sorted, "--", label="pred cl")
plt.xlabel("aoa [deg]")
plt.ylabel("cl")
plt.title("Validation (tuned): aoa vs cl (true vs pred)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "aoa_vs_cl_val_tuned.png"), dpi=150)
plt.close()

# (2) aoa vs cd
plt.figure(figsize=(10, 4))
plt.plot(aoa_sorted, cd_true_sorted, label="true cd")
plt.plot(aoa_sorted, cd_pred_sorted, "--", label="pred cd")
plt.xlabel("aoa [deg]")
plt.ylabel("cd")
plt.title("Validation (tuned): aoa vs cd (true vs pred)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "aoa_vs_cd_val_tuned.png"), dpi=150)
plt.close()

# -------------------------------------------------
# 8) 모델 + 스케일러 저장 (dict 형태)
# -------------------------------------------------
bundle = {
    "model": best_model,
    "X_scaler": X_scaler,
    "Y_scaler": Y_scaler,
}

model_path = os.path.join(outdir, "mini_mlp_model_tuned.joblib")
joblib.dump(bundle, model_path)

print(f"\nSaved tuned model & scalers to: {model_path}")
print(f"Outputs saved in folder: {outdir}")
