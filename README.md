# ============================================================
# PEPA — Plataforma de Economia Petrolífera Angolana
# Script Python completo e funcional (pronto para GitHub)
# Regressão Linear + Random Forest + Conjectura de Collatz
# Dados: Angola 2000-2025 | Previsão: 2027-2033
# Autor: Rogério Bernardo Manuel
# ============================================================
# Instalação: pip install -r requirements.txt
#   numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- 1. DADOS HISTÓRICOS ANGOLA (2000-2025) ----
data = {
    'ano': list(range(2000, 2026)),
    'pib_angola':          [9.1,8.9,11.5,14.1,19.8,28.9,41.7,60.4,84.2,75.5,82.5,104.3,115.3,124.2,126.8,102.6,95.3,122.1,101.4,89.4,53.6,72.5,106.7,84.7,92.3,97.8],
    'preco_barril':        [28.5,24.4,25.0,28.8,38.3,54.6,65.4,72.6,97.3,61.7,79.6,111.3,111.7,108.7,99.0,52.4,44.1,54.7,71.3,64.2,41.8,70.9,99.0,82.5,78.1,74.3],
    'producao_diaria':     [746,742,895,877,989,1244,1428,1693,1900,1784,1767,1660,1728,1711,1716,1767,1722,1632,1505,1382,1224,1124,1175,1120,1100,1080],
    'taxa_cambio':         [9.2,22.4,43.5,75.0,83.5,87.0,80.4,75.0,75.0,79.3,91.9,93.9,95.8,97.0,97.6,135.3,165.9,165.9,308.6,364.8,578.3,631.4,460.6,828.8,880.0,960.0],
    'inflacao':            [325.0,152.6,105.6,98.2,43.6,23.0,13.3,12.2,12.5,13.7,14.5,13.5,10.3,8.8,7.3,10.3,32.4,29.8,19.6,17.1,22.3,25.8,21.4,13.6,11.8,9.4],
    'exportacoes_petroleo':[7.9,6.5,8.2,9.0,13.0,23.0,30.7,44.4,66.3,40.1,47.9,65.8,70.1,68.2,57.8,33.1,27.6,33.6,40.8,34.7,19.0,30.1,43.2,35.8,32.5,30.8],
    'investimento_estrangeiro':[879,2146,1672,3506,1449,-1193,-259,-925,1679,-3289,-3227,-3076,-6788,-6807,1946,-2293,4104,-7397,-5732,-4097,-1866,-3430,-2905,-1500,-1200,850],
    'politica_governamental':[0.3,0.3,0.4,0.4,0.5,0.5,0.6,0.6,0.7,0.4,0.5,0.6,0.6,0.6,0.6,0.4,0.4,0.5,0.5,0.5,0.3,0.5,0.6,0.5,0.5,0.6],
}
df = pd.DataFrame(data)
print("Dataset Angola:", df.shape)
print(df.describe().round(2))

# ---- 2. CONJECTURA DE COLLATZ ----
# Valores admissíveis de n (na PEPA): 3, 10, 21, 64, 128
def collatz_sequence(n):
    seq = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq

def collatz_features(n):
    seq = collatz_sequence(n)
    return {
        'steps': len(seq) - 1,
        'max_value': max(seq),
        'mean_value': sum(seq) / len(seq),
        'sequence': seq,
        'n_even': sum(1 for v in seq if v % 2 == 0),   # par = regulação
        'n_odd': sum(1 for v in seq if v % 2 != 0),    # ímpar = impulso
    }

COLLATZ_N = 10   # 6 passos (2027-2032) | n=64 também: 6 passos
cf = collatz_features(COLLATZ_N)
print(f"\nCollatz({COLLATZ_N}): {cf['steps']} passos, máx={cf['max_value']}")

# ---- 3. PREPARAR FEATURES ----
FEATURES = ['preco_barril', 'producao_diaria', 'taxa_cambio', 'inflacao',
            'exportacoes_petroleo', 'investimento_estrangeiro', 'politica_governamental']
TARGET = 'pib_angola'

X = df[FEATURES].values
y = df[TARGET].values

# Adicionar features Collatz (normalizada + paridade)
seq = cf['sequence']
collatz_col = np.array([seq[i % len(seq)] / cf['max_value'] for i in range(len(X))])
collatz_parity = np.array([1 if v % 2 == 0 else -1 for v in
                           [seq[i % len(seq)] for i in range(len(X))]])
X = np.column_stack([X, collatz_col, collatz_parity])
feature_names = FEATURES + [f'Collatz_norm(n={COLLATZ_N})', f'Collatz_parity(n={COLLATZ_N})']

# ---- 4. TREINO / TESTE (temporal 80/20) ----
n_cut = int(len(X) * 0.8)
X_train, X_test = X[:n_cut], X[n_cut:]
y_train, y_test = y[:n_cut], y[n_cut:]
print(f"\nTreino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

# ---- 5. REGRESSÃO LINEAR ----
lr = LinearRegression().fit(X_train, y_train)
lr_tr, lr_te = lr.predict(X_train), lr.predict(X_test)

# ---- 6. RANDOM FOREST ----
rf = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1).fit(X_train, y_train)
rf_tr, rf_te = rf.predict(X_train), rf.predict(X_test)

# ---- 7. ENSEMBLE (média LR + RF) ----
ens_te = (lr_te + rf_te) / 2

def metricas(nome, ytr, ptr, yte, pte):
    print(f"\n=== {nome} ===")
    print(f"  R2 Treino: {r2_score(ytr, ptr):.4f}")
    print(f"  R2 Teste:  {r2_score(yte, pte):.4f}")
    print(f"  RMSE:      {np.sqrt(mean_squared_error(yte, pte)):.4f}")
    print(f"  MAE:       {mean_absolute_error(yte, pte):.4f}")

metricas("Regressão Linear", y_train, lr_tr, y_test, lr_te)
metricas("Random Forest", y_train, rf_tr, y_test, rf_te)
metricas("Ensemble (LR+RF)", y_train, (lr_tr + rf_tr) / 2, y_test, ens_te)

print("\nImportância das features (RF):")
for name, imp in sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:32s} {imp:.4f}")

# ---- 8. PREVISÃO IEPA/OSEGI 2027-2033 (5 cenários de Collatz) ----
COLLATZ_SCENARIOS = [3, 10, 21, 64, 128]
HORIZON = list(range(2027, 2034))  # 7 anos

def prever(anos, col_n, politica_mult=1.0, politica_val=0.5):
    cfx = collatz_features(col_n)
    seqx = cfx['sequence']
    ultimo = df.iloc[-1]
    tendencias = {}
    for col in FEATURES:
        vals = df[col].values
        tendencias[col] = (vals[-1] - vals[0]) / (len(vals) - 1)
    out = []
    for ano in anos:
        ahead = ano - df['ano'].iloc[-1]
        row = {}
        for col in FEATURES:
            row[col] = politica_val if col == 'politica_governamental' else float(ultimo[col]) + tendencias[col] * ahead
        idx = (ano - 2027) % len(seqx)   # um passo por ano de previsão
        c_val = seqx[idx]
        c_norm = c_val / cfx['max_value']
        row_vec = [row[f] for f in FEATURES] + [c_norm, 1 if c_val % 2 == 0 else -1]
        Xrow = np.array([row_vec])
        lr_p = lr.predict(Xrow)[0]
        rf_p = rf.predict(Xrow)[0]
        ens = (lr_p + rf_p) / 2
        fase = 'Pico' if c_norm > 0.7 else 'Vale' if c_norm < 0.2 else 'Normal'
        out.append({'ano': ano, 'LR': round(lr_p, 2), 'RF': round(rf_p, 2),
                    'Ensemble': round(ens * politica_mult, 2), 'Collatz': c_val, 'Fase': fase})
    return pd.DataFrame(out)

print("\n" + "=" * 62)
print("PREVISÃO IEPA/OSEGI 2027-2033 — CINCO CENÁRIOS DE COLLATZ")
print("=" * 62)
for n_val in COLLATZ_SCENARIOS:
    tab = prever(HORIZON, n_val, politica_mult=1.0)
    growth = (tab['Ensemble'].iloc[-1] - tab['Ensemble'].iloc[0]) / tab['Ensemble'].iloc[0] * 100
    print(f"\nCollatz n={n_val}: crescimento {growth:+.1f}%")
    print(tab.to_string(index=False))

# ---- 9. VISUALIZAÇÃO ----
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('PEPA — Análise do IEPA Angola (LR + RF + Collatz)', fontsize=15,
             color='#cc2222', fontweight='bold')

ax1 = axes[0, 0]
ax1.plot(df['ano'], df['pib_angola'], 'r-o', ms=4, linewidth=2, label='IEPA Histórico')
tab10 = prever(HORIZON, 10, politica_mult=1.0)
ax1.plot(tab10['ano'], tab10['Ensemble'], 'y--s', ms=5, linewidth=2, label='Ensemble 2027-2033')
ax1.axvline(x=2025, color='gray', linestyle=':', alpha=0.8, label='Presente')
ax1.set_title('IEPA — Série Temporal + Previsão', fontsize=11)
ax1.set_xlabel('Ano'); ax1.set_ylabel('Bi USD'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2 = axes[0, 1]
for n_val, cor in zip(COLLATZ_SCENARIOS, ['#e05050', '#e67e22', '#f5c518', '#4ade80', '#50b0e0']):
    t2 = prever(HORIZON, n_val, politica_mult=1.0)
    ax2.plot(t2['ano'], t2['Ensemble'], '-o', ms=3, linewidth=1.6, color=cor, label=f'n={n_val}')
ax2.set_title('Ensemble por Cenário de Collatz', fontsize=11)
ax2.set_xlabel('Ano'); ax2.set_ylabel('Bi USD'); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

ax3 = axes[1, 0]
seq_plot = cf['sequence']
cores = ['#4ade80' if v % 2 == 0 else '#e67e22' for v in seq_plot]
ax3.bar(range(len(seq_plot)), seq_plot, color=cores, alpha=0.85)
ax3.axhline(y=cf['max_value'], color='red', linestyle='--', alpha=0.6, label=f'Pico: {cf["max_value"]}')
ax3.set_title(f'Conjectura de Collatz (n={COLLATZ_N})', fontsize=11)
ax3.set_xlabel('Passo'); ax3.set_ylabel('Valor'); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

ax4 = axes[1, 1]
ax4.scatter(y_test, ens_te, color='#cc2222', alpha=0.85, s=70, edgecolors='white', linewidth=0.5, label='Real vs Previsto')
lim = [min(y_test.min(), ens_te.min()) - 5, max(y_test.max(), ens_te.max()) + 5]
ax4.plot(lim, lim, '--', color='#555', linewidth=1.5, label='Previsão perfeita')
ax4.set_title('Real × Previsto — Ensemble', fontsize=11)
ax4.set_xlabel('IEPA Real (Bi USD)'); ax4.set_ylabel('IEPA Previsto (Bi USD)'); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pepa_results.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
print("\n✅ Gráfico guardado: pepa_results.png")

# ---- 10. EXPORTAR RESULTADOS EM CSV ----
resumo = pd.concat([prever(HORIZON, n_val, politica_mult=1.0).assign(n=n_val) for n_val in COLLATZ_SCENARIOS])
resumo.to_csv('PEPA_Previsao_2027_2033.csv', index=False, encoding='utf-8-sig')
df.to_csv('PEPA_Dados_Historicos.csv', index=False, encoding='utf-8-sig')
print("✅ Ficheiros exportados: PEPA_Previsao_2027_2033.csv | PEPA_Dados_Historicos.csv")
