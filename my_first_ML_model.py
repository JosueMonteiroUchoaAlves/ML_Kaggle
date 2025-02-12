#              Written in Portuguese to familiarize myself                    #
#                       Comentários próprios                                  #
#        sklearn é a biblioteca para criar os modelos de aprendizado          #
#        .tree é porque é o tipo árvore de decisao                            #
#        importo uma decision tree do tipo regressor, que foi o               #                        
#        exemplo do kaggle ver mais sobre os tipos de DT:                     #
#        https://scikit-learn.org/stable/                                     #

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

link_para_database = r'./training_data/train.csv'

dados_das_casas = pd.read_csv(link_para_database)

# O random state garante que vou conseguir o mesmo resultado toda vez
# Cada numero vai ter um resultado diferente
modelo_da_analise = DecisionTreeRegressor(random_state=1)

# Selecionando o meu dado objetivo para fazer a previsao
# que, por convencao, é chamado de 'y'

y = dados_das_casas.SalePrice 

# exibindo colunas
print(dados_das_casas.columns)

# Vamos pegar as features (atributos, caracteristicas) que possam influenciar o preco das casas

colunas_de_interesse = [
  'LotArea',
  'YearBuilt',
  '1stFlrSF',
  '2ndFlrSF',
  'FullBath',
  'BedroomAbvGr',
  'TotRmsAbvGrd',
  'PoolArea', 
  'GarageArea', 
  'LotFrontage', 
  ]

# Selecionando varias colunas por meio de uma lista entre colchetes
X = dados_das_casas[colunas_de_interesse]

#print(X.describe())

# Repartindo dados: treino e teste

# random_state com um valor numerico garante que possamos conseguir as mesmas particoes todas as vezes
treino_X, teste_X, treino_y, teste_y = train_test_split(X, y, random_state=1)

# Ajustando/ treinando o modelo

modelo_da_analise.fit(treino_X, treino_y)

valor_das_previsoes = modelo_da_analise.predict(teste_X)

# Avaliando a eficiência do modelo

print(mean_absolute_error(teste_y, valor_das_previsoes))
