# Regressão múltipla OLM para previsão de consumo de energia elétrica fabril (em kWh) baseado em horas de produção e temperatura (°C)

Este repositório contém um projeto de análise e modelagem de regressão linear para prever o consumo de energia com base em variáveis como `Horas Totais` e `Temperatura Média`. 

## Descrição

O projeto utiliza **Python** e bibliotecas de ciência de dados para analisar uma base de dados e construir modelos de regressão linear. O objetivo é prever o consumo de energia (`Consumo (kWh)`) com base em variáveis independentes que influenciam o consumo. 

O script inclui:
- Carregamento e análise dos dados.
- Visualização das correlações entre as variáveis através de mapas de calor.
- Construção de modelos de regressão linear usando **Scikit-Learn** e **Statsmodels**.
- Avaliação do modelo através de métricas como **MSE** (Mean Squared Error) e **R²**.
- Visualização tridimensional dos dados reais e da superfície de predição usando **Plotly**.

## Estrutura do Código

1. **Instalação dos Pacotes**: O script inclui comandos para instalação dos pacotes necessários.
2. **Importação de Bibliotecas**: Importação das bibliotecas necessárias para manipulação de dados, visualização gráfica, modelagem e avaliação.
3. **Carregamento dos Dados**: Carregamento dos dados do arquivo `Dados Fábrica.xlsx`.
4. **Análise de Correlação**: Visualização das correlações entre as variáveis quantitativas usando um mapa de calor.
5. **Construção do Modelo de Regressão Linear**:
   - Utiliza **Scikit-Learn** para ajustar o modelo e prever o consumo de energia.
   - Utiliza **Statsmodels** para obter estatísticas detalhadas do modelo.
6. **Gráfico 3D**: Geração de um gráfico tridimensional dos pontos reais e da superfície de predição para ilustrar o ajuste do modelo.

## Dependências

Certifique-se de ter as seguintes bibliotecas instaladas:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `plotly`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `pingouin`
- `statstests`

