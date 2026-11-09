# Global Solution - Machine Learning & Advanced Mathematics (MLAM)
**Tema:** CitizenTech – Sustentabilidade Urbana e Qualidade do Ar (US Air Quality)

---

## 📋 Sobre o Projeto
Este repositório contém a entrega da **Global Solution**, desenvolvida no âmbito da disciplina de *Machine Learning & Advanced Mathematics (MLAM)*. O projeto aplica rigor estatístico, probabilidade, inferência, regressões lineares (simples e múltipla) e regressão logística binária sobre dados reais de qualidade do ar da EPA (Environmental Protection Agency) obtidos via Kaggle.

O objetivo central é modelar e prever indicadores de poluição atmosférica (`Median AQI`) com base na contagem de dias com ar limpo e na incidência de poluentes críticos como **PM2.5** e **Ozônio**, subsidiando decisões para cidades inteligentes e sustentáveis (*CitizenTech*).

---

## 👥 Integrante do Grupo
* **Kayo Henderson** — RM: 570706


---

## 📂 Estrutura do Repositório

```text
├── us_air_quality.csv       # Base de dados original (Kaggle/EPA)
├── analise_dados.py         # Script Python completo com todas as análises e modelos
├── relatorio_estatistico.pdf# Relatório executivo detalhado em PDF
└── README.md                # Documentação oficial do projeto
```

---

## 📊 Metodologia e Etapas do Projeto

O projeto está dividido em módulos estatísticos executados via script Python:

1. **Tratamento e Agregação de Dados (`analise_dados.py`):**
   - Leitura da base bruta de qualidade do ar e agrupamento municipal/CBSA.
   - Limpeza e filtragem de valores nulos para assegurar integridade estatística.
2. **Distribuição Normal de Probabilidade:**
   - Cálculo de média, mediana e desvio padrão populacional do índice `Median AQI`.
   - Avaliação de probabilidades via função de distribuição cumulativa (`CDF`).
3. **Testes de Hipóteses para a Média:**
   - Aplicação de testes estatísticos $Z$ para amostras ($n=20$ e $n=25$) sob diferentes níveis de significância ($\alpha = 6\%$ e $\alpha = 9\%$).
4. **Análise Bivariada e Regressão Linear Simples:**
   - Cálculo da covariância e Correlação de Pearson entre dias bons e índice de qualidade.
   - Ajuste de modelo OLS (`statsmodels`) para predição linear do AQI.
5. **Regressão Linear Múltipla:**
   - Construção de matrizes de covariância e correlação multivariada (`Good Days`, `Days PM2.5`, `Days Ozone`).
   - Estimação do modelo OLS múltiplo com avaliação do coeficiente de determinação ($R^2$) e Teste F.
6. **Regressão Logística Binária:**
   - Definição de limiar crítico de qualidade do ar (`AQI > limiar`).
   - Treinamento de classificador logístico (`scikit-learn`), extração da matriz de confusão e métrica de acurácia.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o **Python 3.8+** instalado em sua máquina junto com as bibliotecas necessárias. Você pode instalá-las executando:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```

### Executando a Análise
1. Clone este repositório ou baixe os arquivos em sua máquina.
2. Certifique-se de que o arquivo da base de dados (`us_air_quality.csv`) está na mesma pasta raiz do script.
3. No terminal, execute o script principal:

```bash
python analise_dados.py
```

---

## 📄 Relatório Estatístico
O relatório gerencial completo, contendo a contextualização teórica, interpretação detalhada dos coeficientes das regressões, matrizes de confusão e limitações metodológicas (fatores meteorológicos e topográficos), encontra-se arquivado no documento **`relatorio_estatistico.pdf`** neste repositório.
