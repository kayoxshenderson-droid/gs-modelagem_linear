# Relatório Estatístico - Global Solution MLAM
**Tema:** CitizenTech - Sustentabilidade Urbana e Qualidade do Ar (US Air Quality)

*(Atenção: Salve/exporte este arquivo como PDF para enviar no formato exigido no item 08.c)*

## 1. Contextualização do Problema
Dentro das diretrizes do programa **CitizenTech**, desenhar infraestruturas e políticas públicas urbanas sustentáveis requer monitoramento constante da poluição. Este relatório avalia o impacto de poluentes específicos (PM2.5 e Ozônio) e do histórico anual de "Dias Bons" (`Good Days`) no Índice Geral de Qualidade do Ar (`Median AQI`) de condados americanos. O objetivo é fornecer um pilar analítico para a criação de painéis de saúde pública e emissão de alertas precoces à população.

## 2. Descrição dos Dados e Procedimentos
Os dados foram coletados através do dataset público **"US Air Quality 1980-Present" (via Kaggle/EPA)**. Foram feitas higienizações de nulos nas colunas alvos. Aplicamos Z-score e a função CDF da Normal para as validações probabilísticas, além de Testes de Hipótese para extrair inferences populacionais com amostras (n=20 e n=25). Finalizamos estruturando modelos preditivos: OLS Simples, OLS Múltipla (`statsmodels`) e Regressão Logística (`scikit-learn`).

## 3. Principais Insights e Resultados

### a) Distribuição Normal e Análise Bivariada
- Na **Regressão Simples**, o cruzamento entre "Good Days" e "Median AQI" demonstrou matematicamente a dinâmica da atmosfera local: a equação de regressão mapeou a relação inversa, provando um $p$-valor do Teste F próximo de zero, o que atesta a forte significância do modelo de monitoramento.

### b) Regressão Linear Múltipla
- Inserindo os causadores da poluição (`Days PM2.5` e `Days Ozone`):
A Matriz de Correlação exibiu comportamentos distintos na composição química do ar, ajudando a identificar quais indústrias/fontes de emissão afetam mais a atmosfera local. O $R^2$ obtido consolidou o entendimento de que a visão agregada dos poluentes fornece previsões incisivamente melhores para o planejamento de saúde urbana do que variáveis isoladas.

### c) Regressão Logística Binária
- Utilizou-se o teto de corte $AQI > 50$ (classificação EPA para ar que deixa de ser "Bom"). O modelo Logístico treinado identificou, via **Matriz de Confusão**, uma acurácia robusta. O limiar calculado demonstra que sensores (foco CitizenTech) captando picos curtos de PM2.5 e Ozônio conseguem acionar gatilhos algorítmicos que preveem o risco ambiental antes mesmo do fim do ano fiscal das cidades.

## 4. Conclusões Finais e Limitações
A análise valida que o monitoramento CitizenTech não precisa de centenas de parâmetros custosos; focar em sensores de PM2.5 e Ozônio cobre grande parte da predição de risco de saúde.
**Limitações:** Fatores climáticos como ventos, chuvas contínuas e geografia (cidades em vales vs. cidades litorâneas) afetam drasticamente a dissipação da poluição e não foram incluídos nas equações. Portanto, o $R^2$ das regressões esbarra em uma margem não explicada que pertence puramente à meteorologia e topografia.
