import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def parse_data(file_path):
    data = pd.read_csv(file_path)
    return data

def setup_data(file_path):
    home_data = parse_data(file_path)

    # create target object and call it y
    y = home_data.SalePrice

    # create X (features)
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
                'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]

    # split data into validation and training sets
    # (static random_state ensures we get same split every time)
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    return train_X, val_X, train_y, val_y


def setup_model(train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(random_state=1)
    model.fit(train_X, train_y)
    return model


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


def calculate_tree_size(train_X, val_X, train_y, val_y):
    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
    mae_dict = {}

    # find the ideal tree size from candidate_max_leaf_nodes
    for max_leaf_nodes in candidate_max_leaf_nodes:
        this_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        mae_dict[max_leaf_nodes] = this_mae

    # return the best value of max_leaf_nodes
    return(min(mae_dict, key=mae_dict.get))


def main():
    # get the model set up using the data from the training set
    train_X, val_X, train_y, val_y = setup_data('train.csv')
    model = setup_model(train_X, val_X, train_y, val_y)
    tree_size = calculate_tree_size(train_X, val_X, train_y, val_y)

    # make validation predictions and calculate mean absolute error
    # val_predictions = model.predict(val_X)
    # val_mae = mean_absolute_error(val_predictions, val_y)
    # print("Validation MAE: {:,.0f}".format(val_mae))

    optimized_model = DecisionTreeRegressor(max_leaf_nodes=tree_size, random_state=1)

    home_data = parse_data('train.csv')

    # create target object and call it y
    y = home_data.SalePrice

    # create X (features)
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
                'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]

    # fit the optimized model
    optimized_model.fit(X, y)

    # make predictions which we will submit.
    price_predictions = optimized_model.predict(X.head())
    print(price_predictions)



if __name__ == "__main__":
    main()
