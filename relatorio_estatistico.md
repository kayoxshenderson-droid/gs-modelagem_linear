# Relatório Estatístico - Global Solution MLAM
**Tema:** CitizenTech - Sustentabilidade Urbana e Qualidade do Ar (US Air Quality)

*(Atenção: Salve/exporte este arquivo como PDF para enviar no formato exigido no item 08.c)*

## 1. Contextualização do Problema
Dentro das diretrizes do programa **CitizenTech**, desenhar infraestruturas e políticas públicas urbanas sustentáveis requer monitoramento constante da poluição. Este relatório avalia a relação entre poluentes específicos (PM2.5 e Ozônio), a quantidade de "Dias Bons" (`Good Days`) e o Índice de Qualidade do Ar mediano (`Median AQI`) de áreas metropolitanas (CBSA) dos Estados Unidos. O objetivo é fornecer uma base analítica para painéis de saúde pública e emissão de alertas precoces à população.

## 2. Descrição dos Dados e Procedimentos
Os dados vêm do dataset público **"US Air Quality 1980-Present" (Kaggle/EPA)**: 216.050 medições diárias entre 1980 e 2022, agregadas por área metropolitana em **671 registros**. Para cada área foram calculados o AQI mediano, a contagem de dias com AQI ≤ 50 (`Good Days`) e a contagem de dias em que o PM2.5 e o Ozônio foram o poluente dominante. Registros nulos foram removidos.

Procedimentos: Z-score e função de distribuição acumulada (CDF) da Normal para as probabilidades; testes Z unilaterais para a média com amostras de n=20 (α=6%) e n=25 (α=9%); regressão linear simples e múltipla (OLS, `statsmodels`); regressão logística binária (`scikit-learn`). Os gráficos gerados pelo script estão na pasta `graficos/`.

## 3. Principais Insights e Resultados

### a) Distribuição Normal
- AQI mediano: média **37,21**, mediana **39,00**, desvio-padrão **11,79**.
- P(X < média) = **50,00%** (evento equiprovável, por definição da Normal).
- P(X > mediana) = **43,96%** (evento moderadamente provável). Como a mediana é maior que a média, a distribuição tem leve assimetria à esquerda.

### b) Testes de Hipóteses
- **n=20, α=6%, H1: μ < 37,21:** média amostral 31,25; Z = −2,26; p-valor = 0,0119 → **rejeita H0**.
- **n=25, α=9%, H1: μ > 37,21:** média amostral 33,44; Z = −1,60; p-valor = 0,9450 → **não rejeita H0**.

### c) Análise Bivariada (Good Days × AQI mediano)
- Covariância 279,24 e correlação de Pearson **r = 0,169** (fraca e positiva).
- Equação: **AQI = 33,9882 + 0,0143 × Good Days**. O Teste F é significativo (p = 1,05 × 10⁻⁵), mas o **R² = 0,029** mostra que a variável explica só 2,9% da variação do AQI.
- Previsão: 100 Good Days → AQI estimado de **35,42**.

### d) Regressão Linear Múltipla
- Correlações entre preditoras: Good Days × Ozone = 0,72; Good Days × PM2.5 = 0,44; PM2.5 × Ozone = 0,04.
- Equação: **AQI = 31,9085 − 0,0697 × GoodDays + 0,0754 × PM2.5 + 0,0856 × Ozone**.
- **R² = 0,509** (R² ajustado 0,507) e Teste F com p-valor = 1,64 × 10⁻¹⁰². Todos os coeficientes são significativos.
- Controlando pelos poluentes, o efeito de Good Days passa a ser **negativo**, como esperado: mais dias bons, AQI menor. Os dias com PM2.5 e Ozônio dominantes elevam o AQI.
- Previsão: 100 Good Days, 50 dias PM2.5 e 30 dias Ozônio → AQI previsto de **31,27**.

### e) Regressão Logística Binária
- Variável resposta: 1 se AQI mediano > **35** (432 áreas), 0 caso contrário (239 áreas).
- Equação: **logit(p) = −0,8488 + 0,0007 × PM2.5 + 0,0098 × Ozone**.
- Matriz de confusão: [[184, 55], [74, 358]] → **acurácia de 80,77%** (avaliada na própria base de treino).
- Para 40 dias PM2.5 e 60 dias Ozônio: **classe 0** (probabilidade de risco de 44,29%).

## 4. Conclusões Finais e Limitações
Os modelos com os poluentes dominantes explicam cerca de metade da variação do AQI mediano (R² ≈ 0,51), contra apenas 2,9% do modelo com Good Days sozinho. Isso indica que monitorar PM2.5 e Ozônio, foco dos sensores CitizenTech, traz bastante informação sobre o risco à qualidade do ar.

**Limitações:**
- As variáveis de contagem (`Good Days`, `Days PM2.5`, `Days Ozone`) dependem de quantos dias cada área tem na base, que vai de 1 a 599 registros. Isso explica a correlação positiva contraintuitiva na análise bivariada. Usar proporções (% dos dias) seria mais adequado.
- A acurácia da regressão logística foi medida nos mesmos dados usados no treino. Uma separação treino/teste daria uma estimativa mais realista.
- Fatores climáticos (vento, chuva) e geográficos (vales, litoral) afetam a dispersão da poluição e não foram incluídos nos modelos.
