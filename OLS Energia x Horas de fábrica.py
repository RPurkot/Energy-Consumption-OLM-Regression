# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 00:07:17 2024

@author: ra058832
"""

# In[0.1]: Instalação dos pacotes

!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install playsound
!pip install pingouin
!pip install emojis
!pip install statstests

# In[0.2]: Importação dos pacotes

import pandas as pd  # manipulação de dados em formato de dataframe
import numpy as np  # operações matemáticas
import seaborn as sns  # visualização gráfica
import matplotlib.pyplot as plt  # visualização gráfica
import plotly.graph_objects as go  # gráficos 3D
from scipy.stats import pearsonr  # correlações de Pearson
import statsmodels.api as sm  # estimação de modelos
from statsmodels.iolib.summary2 import summary_col  # comparação entre modelos
from sklearn.model_selection import train_test_split  # divisão dos dados
from sklearn.linear_model import LinearRegression  # modelo de regressão linear
from sklearn.metrics import mean_squared_error, r2_score  # avaliação do modelo

#%% Carregar e visualizar os dados
file_path = 'Dados Fábrica.xlsx'
df = pd.read_excel(file_path, sheet_name='Planilha1')

# Exibir as primeiras linhas do dataframe para validação
df.head()

#%% Análise de Correlação
correlation_matrix = df.iloc[:, 1:4].corr()

# Mapa de calor com as correlações entre todas as variáveis quantitativas
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".4f",
                      cmap=plt.cm.Greens,
                      annot_kws={'size': 25}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=17)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=17)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

#%% Separar as variáveis independentes (X) e a dependente (y)
X = df[['Horas Totais', 'Temperatura média']]
y = df['Consumo (kWh)']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Modelo de Regressão Linear com Scikit-Learn
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Exibir métricas do modelo
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Exibir os coeficientes
print("Coeficientes do modelo:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

#%% Modelo com Statsmodels para obter estatísticas mais detalhadas
# Adicionando uma constante para o intercepto
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()

# Resumo estatístico
print(model_sm.summary())

#%% Gráfico 3D com pontos amostrais e superfície de predição
# Criar gráfico de dispersão 3D dos dados reais
trace = go.Scatter3d(
    x=df['Horas Totais'],
    y=df['Temperatura média'],
    z=df['Consumo (kWh)'],
    mode='markers',
    marker={
        'size': 10,
        'color': 'darkorchid',
        'opacity': 0.7,
    },
)

# Layout do gráfico
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800,
    plot_bgcolor='white',
    scene=dict(
        xaxis_title='Horas Totais',
        yaxis_title='Temperatura média',
        zaxis_title='Consumo (kWh)',
        xaxis=dict(gridcolor='rgb(200, 200, 200)', backgroundcolor='whitesmoke'),
        yaxis=dict(gridcolor='rgb(200, 200, 200)', backgroundcolor='whitesmoke'),
        zaxis=dict(gridcolor='rgb(200, 200, 200)', backgroundcolor='whitesmoke')
    )
)

# Adicionar os pontos amostrais ao gráfico
data = [trace]
plot_figure = go.Figure(data=data, layout=layout)

# Criar uma grade de valores para os eixos X e Y
horas_totais = np.linspace(df['Horas Totais'].min(), df['Horas Totais'].max(), 50)
temperatura_media = np.linspace(df['Temperatura média'].min(), df['Temperatura média'].max(), 50)
horas_totais, temperatura_media = np.meshgrid(horas_totais, temperatura_media)

# Calcular os valores previstos de Consumo (kWh) com base na superfície do modelo
consumo_pred = model.intercept_ + model.coef_[0] * horas_totais + model.coef_[1] * temperatura_media

# Adicionar a superfície ajustada ao gráfico
plot_figure.add_trace(go.Surface(
    x=horas_totais,
    y=temperatura_media,
    z=consumo_pred,
    colorscale='Viridis',
    opacity=0.5,
    name='Superfície Ajustada'
))

# Salvar e abrir o gráfico no navegador
plot_figure.write_html('Projeto_regressao_Consumo_energia_ajustado.html')
import webbrowser
webbrowser.open('Projeto_regressao_Consumo_energia_ajustado.html')