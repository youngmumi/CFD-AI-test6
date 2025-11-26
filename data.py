import os
import pandas as pd
import matplotlib.pyplot as plt

# ====== 1. 파일 경로 ======
csv_path = "original.csv"
outdir = "./output"
os.makedirs(outdir, exist_ok=True)

# ====== 2. CSV 불러오기 ======
df = pd.read_csv(csv_path)

# ====== 3. 컬럼 자동 탐색 ======
def find_col(possible_names):
    for col in df.columns:
        key = col.lower().replace(" ", "")
        if any(key.startswith(name) for name in possible_names):
            return col
    return None

alpha_col = find_col(["alpha", "aoa", "angle"])
cl_col    = find_col(["cl", "lift"])
cd_col    = find_col(["cd", "drag"])

print("Detected Columns → Alpha:", alpha_col, "Cl:", cl_col, "Cd:", cd_col)

# ====== 4. 그래프 저장 함수 ======
def save_lineplot(x, y, xlabel, ylabel, filename, color="blue"):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=1.8, color=color)

    # y축 범위 자동 설정
    ymin = y.min() - abs(y.min()) * 0.1
    ymax = y.max() + abs(y.max()) * 0.1
    plt.ylim(ymin, ymax)

    # 0 기준선 표시 (음수 강조)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel}")
    plt.grid(True)

    save_path = os.path.join(outdir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ Saved:", save_path)

# ====== 5. Alpha–Cl ======
if alpha_col and cl_col:
    save_lineplot(
        df[alpha_col], df[cl_col],
        alpha_col, cl_col,
        "Alpha_Cl_lineplot.png"
    )

# ====== 6. Alpha–Cd ======
if alpha_col and cd_col:
    save_lineplot(
        df[alpha_col], df[cd_col],
        alpha_col, cd_col,
        "Alpha_Cd_lineplot.png",
        color="red"
    )
