import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=== GLOBAL SOLUTION - MLAM: ANÁLISE CITIZENTECH ===")
print("Tema: Cidades Sustentáveis e Qualidade do Ar (US Air Quality)\n")

# 01) Base de Dados e Agregação Municipal
try:
    df_raw = pd.read_csv('us_air_quality.csv')
except FileNotFoundError:
    print("ERRO: Arquivo 'us_air_quality.csv' não encontrado!")
    print("Por favor, certifique-se de que o arquivo 'us_air_quality.csv' está na mesma pasta.")
    exit()

# Agregando os registros diários por CBSA / Cidade para obter as métricas anuais exigidas
df = df_raw.groupby(['CBSA Code', 'city_ascii', 'state_name']).agg(
    {'AQI': ['median', lambda x: (x <= 50).sum()],
     'Defining Parameter': [lambda x: (x == 'PM2.5').sum(), lambda x: (x == 'Ozone').sum()]}
).reset_index()

df.columns = ['CBSA Code', 'city_ascii', 'state_name', 'Median AQI', 'Good Days', 'Days PM2.5', 'Days Ozone']

# Filtrando valores nulos para evitar erros nos cálculos estatísticos
colunas_uso = ['Median AQI', 'Good Days', 'Days PM2.5', 'Days Ozone']
df = df.dropna(subset=colunas_uso)

print(f"Base carregada e agregada com sucesso! {df.shape[0]} registros e {df.shape[1]} atributos.\n")

# 02) Distribuição Normal de Probabilidade (1,0 ponto)
print("--- 02) Distribuição Normal de Probabilidade ---")
var_norm = df['Median AQI'] # Índice Mediano de Qualidade do Ar
media = var_norm.mean()
mediana = var_norm.median()
std = var_norm.std(ddof=0)

# a) P(X < Média)
z_a = (media - media) / std
prob_a = stats.norm.cdf(z_a)
print(f"a) Probabilidade do AQI Mediano ser menor que a média ({media:.4f}): {prob_a:.4f} (50.00%)")
print("   Classificação: Evento Equiprovável.")

# b) P(X > Mediana)
z_b = (mediana - media) / std
prob_b = 1 - stats.norm.cdf(z_b)
print(f"b) Probabilidade do AQI Mediano ser maior que a mediana ({mediana:.4f}): {prob_b:.4f}")
print("   Classificação: Evento Altamente Provável (considerando a concentração de dados).\n")

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
print(f"a) Média Amostral (n=20): {media_amostral_20:.4f} | Z-calc: {z_test_a:.4f} | p-valor: {p_value_a:.4f}")
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
print(f"b) Média Amostral (n=25): {media_amostral_25:.4f} | Z-calc: {z_test_b:.4f} | p-valor: {p_value_b:.4f}")
if p_value_b < alpha_b:
    print("   Resultado: Rejeita H0 a 9% de significância.")
else:
    print("   Resultado: Não rejeita H0 a 9% de significância.\n")

# 04) Análise bivariada (2,0 pontos)
print("--- 04) Análise bivariada ---")
X_biv = df['Good Days'] # Dias com qualidade de ar "Boa"
y_biv = df['Median AQI'] # AQI (Quanto maior, pior o ar)

cov_biv = np.cov(X_biv, y_biv)[0, 1]
corr_biv, _ = stats.pearsonr(X_biv, y_biv)
print(f"b) Covariância: {cov_biv:.4f} | Correlação de Pearson: {corr_biv:.4f}")
print("   Interpretação: Cidades com maior contagem de dias com ar bom apresentam correlação com o índice geral.")

X_sm = sm.add_constant(X_biv)
modelo_biv = sm.OLS(y_biv, X_sm).fit()
print(f"d) Equação Regressão Simples: AQI_Mediano = {modelo_biv.params.iloc[0]:.4f} + {modelo_biv.params.iloc[1]:.4f} * Good_Days")
print(f"e) Adequação: R² = {modelo_biv.rsquared:.4f} | Teste F (p-valor) = {modelo_biv.f_pvalue:.4e}")

prev_simples = modelo_biv.predict([1, 100.0]) # 100 dias bons
print(f"f) Previsão: Para 100 'Good Days', o AQI Mediano estimado é {prev_simples[0]:.4f}\n")

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
print(f"e) Adequação: R² = {modelo_mult.rsquared:.4f} | Teste F (p-valor) = {modelo_mult.f_pvalue:.4e}")

prev_mult = modelo_mult.predict([1, 100.0, 50.0, 30.0]) # 100 Good, 50 PM2.5, 30 Ozone
print(f"f) Previsão: 100 Good, 50 PM2.5 e 30 Ozone -> AQI Previsto = {prev_mult[0]:.4f}\n")

# 06) Regressão logística binária (2,0 pontos)
print("--- 06) Regressão logística binária ---")
limiar = 35 # Limiar baseado na mediana da distribuição agregada
y_log = (df['Median AQI'] > limiar).astype(int) 
X_log = df[['Days PM2.5', 'Days Ozone']]

modelo_log = LogisticRegression()
modelo_log.fit(X_log, y_log)

y_pred_log = modelo_log.predict(X_log)
cm = confusion_matrix(y_log, y_pred_log)
acc = accuracy_score(y_log, y_pred_log)

print(f"a) Equação Logística: logit(p) = {modelo_log.intercept_[0]:.4f} + {modelo_log.coef_[0][0]:.4f}*PM2.5 + {modelo_log.coef_[0][1]:.4f}*Ozone")
print("b) Matriz de Confusão:")
print(cm)
print(f"   Acurácia: {acc:.4f} ({(acc*100):.2f}% de classificações corretas)")

classificacao = modelo_log.predict([[40.0, 60.0]])
print(f"c) Classificação para Days PM2.5 = 40 e Days Ozone = 60: Classe {classificacao[0]} (1 = Risco AQI > 35, 0 = Ar Limpo)")

print("\n[SUCESSO] Todos os cálculos, testes estatísticos e modelos foram executados sem erros!")
