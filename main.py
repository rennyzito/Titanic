# %% [markdown]
# # **Parte 1: Infraestrutura**

# %% [markdown]
# 
# - python 3.9+ ✅
# - miniconda setup ✅
# - requirements.txt ✅
# - git público ✅

# %% [markdown]
# # **Parte 2: Escolha de base de dados e análise expoloratória**

# %% [markdown]
# 
# ## **Base de Dados**
# 
# Base de dados escolhida: Titanic - Machine Learning from Disaster
# Link: https://www.kaggle.com/c/titanic/data
# 
# ## **Objetivo**
# 
# Realizar análise exploratória sobre dados de passageiros do Titanic à fim de entender as correlações entre sobreviventes e mortos.

# %% [markdown]
# ## **Análise Exploratória**

# %% [markdown]
# ### Importação de Bibliotecas e leitura de dados

# %%
# importação de bibliotecas

import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

# %%
# Leitura das bases de dados

db = pd.read_csv('titanic_data/train.csv')
db.head(20)

# %%
db.info()

# %%
db.describe()

# %% [markdown]
# ### Análise da Variável Alvo (y): Entendendo a distribuição de Sobrevivência

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

contagem_y = db['Survived'].value_counts()

porcentagens_y = db['Survived'].value_counts(normalize=True) * 100

df_y = pd.DataFrame({
    'Contagem': contagem_y,
    'Porcentagem': porcentagens_y
})
df_y.index = ['Não Sobreviveu (0)', 'Sobreviveu (1)']

print("Tabela de Sobrevivência (Y):")
print(df_y)

plt.figure(figsize=(7, 5))
sns.barplot(x=df_y.index, y='Porcentagem', data=df_y, palette=['salmon', 'skyblue'])

plt.title('Distribuição da Sobrevivência (Variável Y)', fontsize=14)
plt.xlabel('Status de Sobrevivência', fontsize=12)
plt.ylabel('Porcentagem de Passageiros (%)', fontsize=12)
plt.ylim(0, 100) # Forçar o eixo Y a ir até 100%

for i, percentage in enumerate(df_y['Porcentagem']):
    plt.text(i, percentage + 2, f'{percentage:.2f}%', ha='center', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# O desbalanceamento não é severo (cerca de 62% vs 38%), mas deve ser notado.

# %% [markdown]
# ### Análise das Variáveis Explicativas: Entendendo a distribuição de cada feature

# %% [markdown]
# #### Gênero vs Sobrevivência

# %%
import pandas as pd

tabela_contagem = pd.crosstab(
    index = db['Sex'],       # Linhas: Gênero
    columns = db['Survived'] # Colunas: Sobreviveu (0 ou 1)
)

# Renomear as colunas para melhor leitura
tabela_contagem.columns = ['Não Sobreviveu', 'Sobreviveu']

print("Tabela de Contagem (Gênero x Sobrevivência):")
print(tabela_contagem)

# %%
tabela_porcentagem = pd.crosstab(
    index = db['Sex'],
    columns = db['Survived'],
    normalize = 'index'
) * 100 

tabela_porcentagem.columns = ['% Não Sobreviveu', '% Sobreviveu']
tabela_porcentagem = tabela_porcentagem.round(2)

print("\nTabela de Porcentagem (Taxa de Sobrevivência por Gênero):")
print(tabela_porcentagem)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))

# Usamos o método 'plot(kind='bar')' na tabela de porcentagem para facilitar
tabela_porcentagem['% Sobreviveu'].plot(kind='bar', color=['skyblue', 'salmon'])

plt.title('Taxa de Sobrevivência por Gênero', fontsize=16)
plt.xlabel('Gênero', fontsize=12)
plt.ylabel('Porcentagem de Sobrevivência (%)', fontsize=12)

# Rótulos no eixo X
plt.xticks(ticks=[0, 1], labels=['Mulher', 'Homem'], rotation=0)

# Adicionar a linha do 50% para contexto
plt.axhline(50, color='gray', linestyle='--', alpha=0.7, label='50% Sobrevivência')

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# %% [markdown]
# Essa tabela mostra claramente a correlação mais forte no dataset: 74.25% das mulheres sobreviveram, enquanto apenas 18.89% dos homens sobreviveram. A variável Sex é, portanto, a feature mais relevante para a previsão.

# %% [markdown]
# #### Portão de Embarque, Tarifa e Classe vs Sobrevivência

# %%
# Contagem de valores nulos em 'Embarked'
nulos_embarked = db['Embarked'].isnull().sum()
print(f"Número de valores nulos em 'Embarked': {nulos_embarked}")

# %%
moda_embarked = db['Embarked'].mode()[0]
print(f"A moda (porto mais comum) é: {moda_embarked}")

# Imputação (Substituição) dos valores nulos
db['Embarked'].fillna(moda_embarked, inplace=True)

# Verificação (deve retornar 0)
print(f"Número de nulos após imputação: {db['Embarked'].isnull().sum()}")

# %%
# 1. Tabela de Porcentagem (Taxa de Sobrevivência por Porto)
tabela_porcentagem_embarked = pd.crosstab(
    index=db['Embarked'],
    columns=db['Survived'],
    normalize='index' # Normaliza por linha (Porto de Embarque)
) * 100

# 2. Renomear e formatar
tabela_porcentagem_embarked.columns = ['% Não Sobreviveu', '% Sobreviveu']
tabela_porcentagem_embarked = tabela_porcentagem_embarked.round(2)

print("\nTaxa de Sobrevivência por Porto de Embarque:")
print(tabela_porcentagem_embarked)

# %%
pd.crosstab(db['Embarked'], db['Pclass'], normalize='index')

# %%
# 1. Tabela de Porcentagem (Taxa de Sobrevivência por Tarifa)
tabela_porcentagem_embarked = pd.crosstab(
    index=db['Fare'],
    columns=db['Survived'],
    normalize='index' 
) * 100

# 2. Renomear e formatar
tabela_porcentagem_embarked.columns = ['% Não Sobreviveu', '% Sobreviveu']
tabela_porcentagem_embarked = tabela_porcentagem_embarked.round(2)

print("\nTaxa de Sobrevivência por Tarifa:")
print(tabela_porcentagem_embarked)

# %%
median_fare = db['Fare'].median()
db['Fare'].fillna(median_fare, inplace=True)

# 2. Configuração e Geração do Histograma
plt.figure(figsize=(10, 6))

sns.histplot(
    data=db, 
    x='Fare', 
    hue='Survived', 
    multiple='stack', 
    kde=True,         
    palette={0: 'salmon', 1: 'skyblue'},
    edgecolor='black'
)

plt.title('Distribuição de Tarifa por Status de Sobrevivência (Titanic)', fontsize=16)
plt.xlabel('Tarifa', fontsize=12)
plt.ylabel('Contagem de Passageiros', fontsize=12)
plt.legend(title='Sobreviveu', labels=['Sim (1)', 'Não (0)'])

plt.show()

# %% [markdown]
# **Conclusão da Análise:**
# O DataFrame resultante permite a seguinte observação crucial:
# 
# Cherbourg (C) tem a taxa de sobrevivência mais alta (55.36%).
# 
# Queenstown (Q) e Southampton (S) têm taxas de sobrevivência significativamente menores (cerca de 31%-33%).
# 
# Justificativa para Relevância: A diferença é grande, o que sugere que o porto de embarque é uma feature relevante. Isso é geralmente uma correlação indireta, pois passageiros de Cherbourg (C) tinham uma proporção maior na Primeira Classe (Pclass=1), do que os outros portos. 
# 
# Além disos, tarifas mais altas estão correlacionadas com Pclass, e com maior sobrevivência.
# 

# %% [markdown]
# #### Idade vs Sobrevivencia

# %%
median_age = db['Age'].median()
db['Age'].fillna(median_age, inplace=True)

# 2. Configuração e Geração do Histograma
plt.figure(figsize=(10, 6))

# Histograma com Seaborn, usando 'hue' para separar Sobreviveu (1) e Não Sobreviveu (0)
sns.histplot(
    data=db, 
    x='Age', 
    hue='Survived', 
    multiple='stack', # Empilha as contagens para visualização clara
    kde=True,         # Adiciona a Curva de Estimativa de Densidade (KDE)
    palette={0: 'salmon', 1: 'skyblue'},
    edgecolor='black'
)

plt.title('Distribuição de Idade por Status de Sobrevivência (Titanic)', fontsize=16)
plt.xlabel('Idade', fontsize=12)
plt.ylabel('Contagem de Passageiros', fontsize=12)
plt.legend(title='Sobreviveu', labels=['Sim (1)', 'Não (0)'])

plt.show()

# %% [markdown]
# # **Parte 3: Clusterização**

# %% [markdown]
# ## Preparação e Pré-processamento dos Dados

# %% [markdown]
# Para que K-Means e DBSCAN funcionem corretamente, todas as features precisam ser numéricas e estar na mesma escala.

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
from warnings import filterwarnings

filterwarnings('ignore') # Ocultar warnings de imputação/scaling

df = pd.read_csv('titanic_data/train.csv')

# 1. Feature Engineering: Extrair Título
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# 2. Feature Engineering: Tamanho da Família e Sozinho
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)

# 3. Selecionar Features para Clusterização (Removendo ID, Nome, Cabine e Survived)
features_para_cluster = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
X = df[features_para_cluster].copy()

# 4. Criação do Pipeline de Pré-processamento
# Numéricas: Imputar pela Mediana e Escalar
numerical_features = ['Age', 'Fare', 'FamilySize']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categóricas: Imputar pela Moda e One-Hot Encode
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 5. Combinar transformações
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# 6. Aplicar o Pré-processamento e obter o array de dados limpos (X_scaled)
X_scaled = preprocessor.fit_transform(X)

# O X_scaled agora está pronto para a clusterização.
print(f"Shape do Dataset pronto para clusterização: {X_scaled.shape}")

# %% [markdown]
# ## K-Means: Índice de Silhueta e Escolha do Número de Clusters

# %% [markdown]
# **Mensuração do Índice de Silhueta**
# 
# O Índice de Silhueta (Silhouette Score) mede quão similar um objeto é ao seu próprio cluster (coesão) em comparação com outros clusters (separação). 
# O valor varia de -1 a +1:
# 
# * +1: Indica que a amostra está bem pareada ao seu próprio cluster e distante dos clusters vizinhos.
# * 0: Indica que a amostra está na fronteira de decisão entre dois clusters.
# * -1: Indica que a amostra foi atribuída ao cluster errado.
# 
# Para escolher o $k$ ideal, calculamos a Silhueta média para vários valores de $k$ e escolhemos o $k$ que maximiza o score.

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Testar k de 2 a 10
range_n_clusters = range(2, 11)
silhouette_avg = []

for n_clusters in range_n_clusters:
    # 1. Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # 2. Calcular o Score de Silhueta
    silhouette_avg.append(silhouette_score(X_scaled, cluster_labels))

# Plotar o resultado
plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, silhouette_avg, marker='o', linestyle='--')
plt.title("Índice de Silhueta para K-Means")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Score Médio de Silhueta")
plt.grid(True)
plt.savefig('silhouette_kmeans.png')
plt.show()

# Determinar o k ótimo:
optimal_k = range_n_clusters[np.argmax(silhouette_avg)]
max_silhouette = np.max(silhouette_avg)
print(f"O número ótimo de clusters (k) para K-Means é: {optimal_k} (Score: {max_silhouette:.3f})")

# 3. Treinar K-Means com o k ótimo
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_optimal.fit_predict(X_scaled)

# %% [markdown]
# ## DBSCAN: Clusterização e Silhueta

# %% [markdown]
# O DBSCAN não requer que o número de clusters seja predefinido ($k$).
# Em vez disso, ele é sensível a dois parâmetros:
# * $\epsilon$ (eps): Distância máxima entre as amostras para uma ser considerada vizinha da outra.
# * min_samples: O número de vizinhos em um raio $\epsilon$ para que um ponto seja considerado um ponto central (core point).
# 
# Ajustar $\epsilon$ é crucial. Tipicamente, $\epsilon$ é escolhido observando o gráfico k-distância (distância até o $k$-ésimo vizinho).

# %%
from sklearn.cluster import DBSCAN

# Escolha de hiperparâmetros (ajustáveis com base em tentativa e erro ou k-distância)
eps_value = 0.9  # Típico para dados padronizados (0.8 - 1.2)
min_samples_value = 5 

# 1. Aplicar DBSCAN
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Contar o número de clusters encontrados (excluindo ruído = -1)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\nDBSCAN Resultados (eps={eps_value}, min_samples={min_samples_value}):")
print(f"Número de clusters encontrados: {n_clusters_dbscan}")
print(f"Número de pontos de ruído (label -1): {n_noise}")

# 2. Calcular o Score de Silhueta (apenas se houver mais de 1 cluster)
if n_clusters_dbscan > 1:
    # A Silhueta pode ser calculada mesmo com pontos de ruído (label -1)
    silhouette_dbscan = silhouette_score(X_scaled, dbscan_labels)
    print(f"Score de Silhueta do DBSCAN: {silhouette_dbscan:.3f}")
else:
    silhouette_dbscan = 0
    print("DBSCAN não formou clusters suficientes para calcular o Score de Silhueta.")

# %% [markdown]
# ## Outras Medidas de Validação

# %% [markdown]
# Para complementar a Silhueta (que é uma medida interna), vamos usar mais duas:
# 
# * **Calinski-Harabasz Index (CH):** Mede a razão entre a dispersão inter-cluster e a dispersão intra-cluster. Valores maiores são melhores.
# * **Davies-Bouldin Index (DB):** Mede a similaridade média entre clusters, onde a similaridade é a razão entre a distância intra-cluster e a distância entre clusters. Valores menores são melhores.

# %%
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# 1. K-Means (usando o k ótimo encontrado)
ch_kmeans = calinski_harabasz_score(X_scaled, kmeans_labels)
db_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)

# 2. DBSCAN (usando os labels encontrados)
# CH e DB requerem mais de um cluster e que o ruído (-1) seja removido ou tratado
# Criamos máscaras para remover o ruído do DBSCAN para estas métricas
mask = (dbscan_labels != -1)
if np.sum(mask) > 0 and len(np.unique(dbscan_labels[mask])) > 1:
    ch_dbscan = calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])
    db_dbscan = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
else:
    ch_dbscan = np.nan
    db_dbscan = np.nan

print("\n--- Medidas de Validação ---")
print(f"K-Means (k={optimal_k}):")
print(f"  Calinski-Harabasz Score (Maior é melhor): {ch_kmeans:.2f}")
print(f"  Davies-Bouldin Score (Menor é melhor): {db_dbscan:.2f}")
print(f"\nDBSCAN:")
print(f"  Calinski-Harabasz Score (Maior é melhor): {ch_dbscan:.2f}")
print(f"  Davies-Bouldin Score (Menor é melhor): {db_dbscan:.2f}")

# %% [markdown]
# ## Conclusão

# %% [markdown]
# # **Parte 4: Medidas de Similaridade**

# %% [markdown]
# ## Um determinado problema, apresenta 10 séries temporais distintas. Gostaríamos de agrupá-las em 3 grupos, de acordo com um critério de similaridade, baseado no valor máximo de correlação cruzada entre elas. Descreva em tópicos todos os passos necessários.

# %% [markdown]
# **Agrupamento por Correlação Cruzada Máxima (Max Cross-Correlation - MCC)**
# 
# O objetivo é agrupar as 10 séries temporais em 3 clusters ($k=3$), utilizando a MCC como critério de similaridade.
# 
# 1. Pré-processamento e Normalização
# * Garantir Uniformidade: Assegurar que todas as séries temporais possuam a mesma frequência e o mesmo número de observações.
# * Normalização: Aplicar a padronização (Z-Score) ou normalização (Min-Max) em todas as séries para que as diferenças de magnitude não influenciem indevidamente a correlação.
# * Estacionariedade: Remover tendências ou sazonalidade (ex: aplicando diferenciação) se a correlação cruzada for muito sensível à não-estacionariedade dos dados.
# 
# 2. Cálculo da Matriz de Similaridade
# * Correlação Cruzada: Para cada par de séries temporais ($TS_A$ e $TS_B$), calcular a correlação cruzada ao longo de uma faixa definida de lags (deslocamentos temporais).
# * Identificação do Máximo (MCC): O critério de similaridade $Similaridade(A, B)$ é definido como o valor máximo da correlação cruzada.
# * Construção da Matriz de Dissimilaridade: Transformar a similaridade em distância/dissimilaridade (para algoritmos baseados em distância):$$\text{Dissimilaridade}(A, B) = 1 - \text{Max}(\text{Correlação Cruzada}(A, B))$$
# 
# 3. Clusterização e ValidaçãoMatriz de Distâncias: Utilizar a matriz de dissimilaridade calculada no passo anterior.
# * Algoritmo de Clusterização: Aplicar um algoritmo capaz de trabalhar com matrizes de pré-distância (ex: Clusterização Hierárquica).
# * Validação:Visualização: Utilizar Dendrogramas (no caso da Hierárquica) para inspecionar visualmente a fusão dos grupos.
# * Métricas: Aplicar métricas de coesão e separação que sejam robustas, como o Índice de Silhueta, diretamente na matriz de dissimilaridade.

# %% [markdown]
# ## Para o problema da questão anterior, indique qual algoritmo de clusterização você usaria. Justifique.

# %% [markdown]
# **Algoritmo Indicado:** Clusterização Hierárquica (Hierarchical Clustering)
# 
# **Justificativa:** 
# * Baseado em Distância: O algoritmo hierárquico é robusto e funciona perfeitamente quando alimentado com uma matriz de distância pré-calculada (a matriz de Dissimilaridade da MCC).
# * Distâncias Não-Euclidianas: Não assume a forma convexa e esférica dos clusters (como o K-Means), o que é vital quando se usa uma métrica de similaridade não-euclidiana como a MCC.
# * Visualização e Flexibilidade: O dendrograma oferece uma ferramenta de validação visual poderosa, permitindo justificar o corte em $k=3$ grupos.

# %% [markdown]
# ## Indique um caso de uso para essa solução projetada.

# %% [markdown]
# O uso da MCC é ideal para problemas onde a forma do padrão é importante, mas o tempo de ocorrência é variável.
# 
# **Caso de Uso Sugerido:**
# Agrupamento de Séries Temporais de Consumo de Energia Diário em Residências.
# 
# **Cenário:** Identificar perfis de uso semelhantes (ex: "família que acorda cedo" vs. "trabalhadores noturnos").
# 
# **Aplicação:** Se uma residência atinge o pico de consumo às 7h e outra atinge um pico semelhante, mas às 8h (um deslocamento de 1 lag), a MCC será alta, agrupando-as no mesmo perfil.
# 
# **Benefício:** Permite que a concessionária crie 3 grupos de perfis para otimizar preços ou gerenciar a demanda na rede.

# %% [markdown]
# ## Sugira outra estratégia para medir a similaridade entre séries temporais. Descreva em tópicos os passos necessários.

# %% [markdown]
# A alternativa mais poderosa para medir similaridade em séries temporais, que varia em velocidade e não apenas em deslocamento temporal linear, é o **Dynamic Time Warping (DTW)**.
# 
# **Estratégia Alternativa: Dynamic Time Warping (DTW)**
# 
# O DTW calcula a distância de desalinhamento, permitindo que os padrões sejam esticados ou comprimidos no tempo para encontrar o caminho de menor custo entre duas séries.

# %% [markdown]
# **Passos Necessários com DTW:**
# 
# 1. **Cálculo da Distância DTW:** Para cada par de séries temporais ($TS_A$ e $TS_B$), calcular o custo mínimo de alinhamento (a distância DTW).
# 2. **Formação da Matriz de Distâncias:** Construir uma matriz simétrica $10 \times 10$ onde cada elemento $(i, j)$ é a Distância DTW entre $TS_i$ e $TS_j$.
# 3. **Clusterização:** Aplicar um algoritmo de agrupamento (como a Clusterização Hierárquica ou Particionamento em Torno de Medoids - PAM) diretamente na Matriz de Distância DTW.
# 4. **Validação e Interpretação:** Validar os agrupamentos e interpretar os clusters resultantes com base nas formas típicas de cada grupo.

# %% [markdown]
# ## Referências


