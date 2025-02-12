from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Since that's not the first time I've done this exact code, I'll write in English.

print("MAE stands for mean absolute error")

def test_model(train_X, val_X, train_Y, val_Y, max_leaf_nodes_variable):
  learner_model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=max_leaf_nodes_variable)

  learner_model.fit(train_X, train_Y)
  predicted_values = learner_model.predict(val_X)

  #return "The MAE for {} is {}".format( max_leaf_nodes_variable , mean_absolute_error(val_Y, predicted_values))
  return mean_absolute_error(val_Y, predicted_values)
database_link = r'./training_data/train.csv'
houses_data = pd.read_csv(database_link)

features_columns = [
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

X = houses_data[features_columns]
y = houses_data.SalePrice 

t_X, v_X, t_y, v_y = train_test_split(X, y, random_state=1)

""" 
The model mustn't have too many leaves because it can overfit the data. 
This means the model is overfitted to the training data, capturing any 
noise it has made and not generalizing it to any other. 
We also cannot have too few leaves so as not to make the model too generic. 
So, let's test the model with different numbers of leaves,  trying to figure
out the sweet spot between underfitting and overfitting.
"""
candidate_max_leaf_nodes = [5, 10, 50, 100, 200, 300, 400, 500, 1000, 5000]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
most_accurate_value = [-1, float('inf')]

for value in candidate_max_leaf_nodes:
  result_value = float(test_model(t_X, v_X, t_y, v_y, value))
  print(result_value)
  most_accurate_value = [value, result_value] if result_value < most_accurate_value[1] else most_accurate_value

print(f"The smallest MAE is: {most_accurate_value[1]:,.2f} with {most_accurate_value[0]} leaves.")
