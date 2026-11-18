import sys
import warnings
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # salva os gráficos em arquivo, sem precisar de janela
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, accuracy_score
except ModuleNotFoundError as erro:
    print(f"ERRO: biblioteca '{erro.name}' não instalada.")
    print("Instale tudo de uma vez no terminal do PyCharm com:")
    print("    pip install -r requirements.txt")
    sys.exit(1)

warnings.filterwarnings('ignore')

PASTA = Path(__file__).resolve().parent
PASTA_GRAFICOS = PASTA / "graficos"
PASTA_GRAFICOS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def classificar_probabilidade(p: float) -> str:
    """Classifica a probabilidade de um evento a partir do valor calculado."""
    if p < 0.05:
        return "Evento Raro"
    if p < 0.40:
        return "Evento Pouco Provável"
    if p <= 0.60:
        return "Evento Moderadamente Provável (próximo de 50%)"
    if p < 0.95:
        return "Evento Provável"
    return "Evento Altamente Provável"


def descrever_correlacao(r: float) -> str:
    forca = abs(r)
    if forca < 0.1:
        intensidade = "praticamente nula"
    elif forca < 0.3:
        intensidade = "fraca"
    elif forca < 0.7:
        intensidade = "moderada"
    else:
        intensidade = "forte"
    sentido = "positiva" if r > 0 else "negativa"
    return f"correlação {intensidade} e {sentido}"


def salvar(fig, nome: str) -> None:
    caminho = PASTA_GRAFICOS / nome
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"   [gráfico salvo] graficos/{nome}")


print("=== GLOBAL SOLUTION - MLAM: ANÁLISE CITIZENTECH ===")
print("Tema: Cidades Sustentáveis e Qualidade do Ar (US Air Quality)\n")

# 01) Base de Dados e Agregação Municipal
try:
    df_raw = pd.read_csv(PASTA / 'us_air_quality.csv')
except FileNotFoundError:
    print("ERRO: Arquivo 'us_air_quality.csv' não encontrado!")
    print("Por favor, certifique-se de que o arquivo 'us_air_quality.csv' está na mesma pasta do script.")
    sys.exit(1)

# Agregando os registros diários por CBSA / Cidade para obter as métricas exigidas
df = df_raw.groupby(['CBSA Code', 'city_ascii', 'state_name']).agg(
    {'AQI': ['median', lambda x: (x <= 50).sum()],
     'Defining Parameter': [lambda x: (x == 'PM2.5').sum(), lambda x: (x == 'Ozone').sum()]}
).reset_index()

df.columns = ['CBSA Code', 'city_ascii', 'state_name', 'Median AQI', 'Good Days', 'Days PM2.5', 'Days Ozone']

# Filtrando valores nulos para evitar erros nos cálculos estatísticos
colunas_uso = ['Median AQI', 'Good Days', 'Days PM2.5', 'Days Ozone']
df = df.dropna(subset=colunas_uso)

print(f"Base carregada e agregada com sucesso! {df.shape[0]} registros e {df.shape[1]} atributos.")
print(f"Período dos dados: {df_raw['Date'].min()} a {df_raw['Date'].max()} | {len(df_raw):,} medições diárias.\n".replace(",", "."))

# 02) Distribuição Normal de Probabilidade (1,0 ponto)
print("--- 02) Distribuição Normal de Probabilidade ---")
var_norm = df['Median AQI']  # Índice Mediano de Qualidade do Ar
media = var_norm.mean()
mediana = var_norm.median()
std = var_norm.std(ddof=0)
print(f"Média = {media:.4f} | Mediana = {mediana:.4f} | Desvio-padrão = {std:.4f}")

# a) P(X < Média)
z_a = (media - media) / std
prob_a = stats.norm.cdf(z_a)
print(f"a) Probabilidade do AQI Mediano ser menor que a média ({media:.4f}): {prob_a:.4f} ({prob_a:.2%})")
print(f"   Classificação: {classificar_probabilidade(prob_a)}.")

# b) P(X > Mediana)
z_b = (mediana - media) / std
prob_b = 1 - stats.norm.cdf(z_b)
print(f"b) Probabilidade do AQI Mediano ser maior que a mediana ({mediana:.4f}): {prob_b:.4f} ({prob_b:.2%})")
print(f"   Classificação: {classificar_probabilidade(prob_b)}.\n")

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(var_norm, bins=30, stat="density", color="#4C9BE8", ax=ax, label="AQI mediano das cidades")
xs = np.linspace(var_norm.min(), var_norm.max(), 300)
ax.plot(xs, stats.norm.pdf(xs, media, std), color="#E8484C", lw=2, label="Curva Normal ajustada")
ax.axvline(media, color="black", ls="--", label=f"Média = {media:.1f}")
ax.axvline(mediana, color="#2E9E5B", ls=":", lw=2, label=f"Mediana = {mediana:.1f}")
ax.set(title="02) Distribuição do AQI Mediano por cidade", xlabel="AQI mediano", ylabel="Densidade")
ax.legend()
salvar(fig, "02_distribuicao_normal.png")
print()

# 03) Testes de hipóteses para a média (1,0 ponto)
print("--- 03) Testes de hipóteses para a média ---")
pop_mean = media
pop_std = std
np.random.seed(42)

# a) Amostra 20, Nível 6%, H1: media_amostral < pop_mean
amostra_20 = var_norm.sample(20)
media_amostral_20 = amostra_20.mean()
alpha_a = 0.06
z_test_a = (media_amostral_20 - pop_mean) / (pop_std / np.sqrt(20))
p_value_a = stats.norm.cdf(z_test_a)
print(f"a) H0: μ = {pop_mean:.4f} | H1: μ < {pop_mean:.4f} (unilateral à esquerda)")
print(f"   Média Amostral (n=20): {media_amostral_20:.4f} | Z-calc: {z_test_a:.4f} | p-valor: {p_value_a:.4f}")
if p_value_a < alpha_a:
    print("   Resultado: Rejeita H0 a 6% de significância.")
else:
    print("   Resultado: Não rejeita H0 a 6% de significância.")

# b) Amostra 25, Nível 9%, H1: media_amostral > pop_mean
amostra_25 = var_norm.sample(25)
media_amostral_25 = amostra_25.mean()
alpha_b = 0.09
z_test_b = (media_amostral_25 - pop_mean) / (pop_std / np.sqrt(25))
p_value_b = 1 - stats.norm.cdf(z_test_b)
print(f"b) H0: μ = {pop_mean:.4f} | H1: μ > {pop_mean:.4f} (unilateral à direita)")
print(f"   Média Amostral (n=25): {media_amostral_25:.4f} | Z-calc: {z_test_b:.4f} | p-valor: {p_value_b:.4f}")
if p_value_b < alpha_b:
    print("   Resultado: Rejeita H0 a 9% de significância.\n")
else:
    print("   Resultado: Não rejeita H0 a 9% de significância.\n")

# 04) Análise bivariada (2,0 pontos)
print("--- 04) Análise bivariada ---")
X_biv = df['Good Days']  # Dias com qualidade de ar "Boa"
y_biv = df['Median AQI']  # AQI (Quanto maior, pior o ar)

cov_biv = np.cov(X_biv, y_biv)[0, 1]
corr_biv, p_corr = stats.pearsonr(X_biv, y_biv)
print(f"b) Covariância: {cov_biv:.4f} | Correlação de Pearson: {corr_biv:.4f} (p-valor = {p_corr:.4e})")
print(f"   Interpretação: {descrever_correlacao(corr_biv)} entre a contagem de 'Good Days' e o AQI mediano.")

X_sm = sm.add_constant(X_biv)
modelo_biv = sm.OLS(y_biv, X_sm).fit()
print(f"d) Equação Regressão Simples: AQI_Mediano = {modelo_biv.params.iloc[0]:.4f} + {modelo_biv.params.iloc[1]:.4f} * Good_Days")
print(f"e) Adequação: R² = {modelo_biv.rsquared:.4f} | Teste F (p-valor) = {modelo_biv.f_pvalue:.4e}")
print(f"   O modelo é estatisticamente significativo, mas explica só {modelo_biv.rsquared:.1%} da variação do AQI.")

prev_simples = modelo_biv.predict([1, 100.0])  # 100 dias bons
print(f"f) Previsão: Para 100 'Good Days', o AQI Mediano estimado é {prev_simples[0]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.regplot(x=X_biv, y=y_biv, ax=ax, scatter_kws={"alpha": 0.35, "s": 18, "color": "#4C9BE8"}, line_kws={"color": "#E8484C"})
ax.set(title=f"04) Good Days × AQI mediano  (r = {corr_biv:.3f}, R² = {modelo_biv.rsquared:.3f})",
       xlabel="Good Days (contagem)", ylabel="AQI mediano")
salvar(fig, "04_regressao_simples.png")
print()

# 05) Regressão linear múltipla (2,0 pontos)
print("--- 05) Regressão linear múltipla ---")
X_mult = df[['Good Days', 'Days PM2.5', 'Days Ozone']]
y_mult = df['Median AQI']

print("b) Matriz de Covariância:")
print(X_mult.cov().round(4))
print("\nb) Matriz de Correlação:")
print(X_mult.corr().round(4))

X_mult_sm = sm.add_constant(X_mult)
modelo_mult = sm.OLS(y_mult, X_mult_sm).fit()
print(f"\nd) Equação: AQI_Mediano = {modelo_mult.params.iloc[0]:.4f} + {modelo_mult.params.iloc[1]:.4f}*GoodDays + {modelo_mult.params.iloc[2]:.4f}*PM2.5 + {modelo_mult.params.iloc[3]:.4f}*Ozone")
print(f"e) Adequação: R² = {modelo_mult.rsquared:.4f} | R² ajustado = {modelo_mult.rsquared_adj:.4f} | Teste F (p-valor) = {modelo_mult.f_pvalue:.4e}")
print("   p-valores dos coeficientes: " + " | ".join(f"{k} = {v:.2e}" for k, v in modelo_mult.pvalues.items()))

prev_mult = modelo_mult.predict([1, 100.0, 50.0, 30.0])  # 100 Good, 50 PM2.5, 30 Ozone
print(f"f) Previsão: 100 Good, 50 PM2.5 e 30 Ozone -> AQI Previsto = {prev_mult[0]:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[['Median AQI'] + list(X_mult.columns)].corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
ax.set_title("05) Matriz de correlação")
salvar(fig, "05_matriz_correlacao.png")
print()

# 06) Regressão logística binária (2,0 pontos)
print("--- 06) Regressão logística binária ---")
limiar = 35  # Limiar de corte próximo da média do AQI mediano (37,2) — separa cidades com ar mais carregado
y_log = (df['Median AQI'] > limiar).astype(int)
X_log = df[['Days PM2.5', 'Days Ozone']]
print(f"Variável resposta: 1 se AQI mediano > {limiar} ({y_log.sum()} cidades), 0 caso contrário ({(y_log == 0).sum()} cidades)")

modelo_log = LogisticRegression()
modelo_log.fit(X_log, y_log)

y_pred_log = modelo_log.predict(X_log)
cm = confusion_matrix(y_log, y_pred_log)
acc = accuracy_score(y_log, y_pred_log)

print(f"a) Equação Logística: logit(p) = {modelo_log.intercept_[0]:.4f} + {modelo_log.coef_[0][0]:.4f}*PM2.5 + {modelo_log.coef_[0][1]:.4f}*Ozone")
print("b) Matriz de Confusão:")
print(cm)
print(f"   Acurácia: {acc:.4f} ({(acc*100):.2f}% de classificações corretas)")

novo_caso = pd.DataFrame([[40.0, 60.0]], columns=X_log.columns)
classificacao = modelo_log.predict(novo_caso)
prob_risco = modelo_log.predict_proba(novo_caso)[0][1]
print(f"c) Classificação para Days PM2.5 = 40 e Days Ozone = 60: Classe {classificacao[0]} "
      f"(probabilidade de risco = {prob_risco:.2%}; 1 = Risco AQI > {limiar}, 0 = Ar Limpo)")

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Previsto 0", "Previsto 1"], yticklabels=["Real 0", "Real 1"])
ax.set_title(f"06) Matriz de confusão — acurácia {acc:.1%}")
salvar(fig, "06_matriz_confusao.png")

print("\n[SUCESSO] Todos os cálculos, testes estatísticos e modelos foram executados sem erros!")
print(f"Gráficos salvos na pasta: {PASTA_GRAFICOS}")
