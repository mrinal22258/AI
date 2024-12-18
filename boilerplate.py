#############
## Imports ##
#############

import pickle
import pandas as pd
import bnlearn as bn
import time
from test_model import test_model


# ######################
# ## Boilerplate Code ##
# ######################

class BayesianNetworkManager:
    """Class to encapsulate Bayesian Network operations."""

    load_data = staticmethod(lambda: (
        pd.read_csv("./Bayesian_Question/train_data.csv"),
        pd.read_csv("./Bayesian_Question/validation_data.csv")
    ))

    create_network = staticmethod(
        lambda df: bn.parameter_learning.fit(
            bn.structure_learning.fit(df, methodtype='hc'), df
        )
    )

    create_pruned_network = staticmethod(lambda df: (
        lambda DAG, adj_matrix: bn.parameter_learning.fit(
            bn.make_DAG([
                (source, target)
                for source in adj_matrix.columns
                for target in adj_matrix.index
                if adj_matrix.loc[source, target] == 1
            ]),
            df
        )
    )(
        DAG := bn.structure_learning.fit(df, methodtype='hc'),
        DAG['adjmat']
    ))

    create_optimized_network = staticmethod(
        lambda df: bn.parameter_learning.fit(
            bn.structure_learning.fit(df, methodtype='hc', black_list=None, white_list=None), df
        )
    )

    save_model = staticmethod(
        lambda fname, model: pickle.dump(model, open(fname, 'wb'))
    )

    load_and_evaluate_model = staticmethod(
        lambda model_name, val_df: test_model(
            pickle.load(open(f"{model_name}.pkl", 'rb')), val_df
        )
    )


def load_data():
    return BayesianNetworkManager.load_data()


def make_network(df):
    model = BayesianNetworkManager.create_network(df)
    # bn.plot(model)
    return model


def make_pruned_network(df):
    model = BayesianNetworkManager.create_pruned_network(df)

    # bn.plot(model)
    return model

def make_optimized_network(df):
    model =  BayesianNetworkManager.create_optimized_network(df)
    # bn.plot(model)
    return model

def save_model(fname, model):
    BayesianNetworkManager.save_model(fname, model)


def evaluate(model_name, val_df):
    correct_predictions, total_cases, accuracy = BayesianNetworkManager.load_and_evaluate_model(model_name, val_df)
    print(f"Total Test Cases: {total_cases}")
    print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
    print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    print("[+] Creating Base Model...")
    start_time = time.time()
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)
    print(f"Base Model created in {time.time() - start_time:.2f} seconds.")

    # Create and save pruned model
    print("[+] Creating Pruned Model...")
    start_time = time.time()
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)
    print(f"Pruned Model created in {time.time() - start_time:.2f} seconds.")

    # Create and save optimized model
    print("[+] Creating Optimized Model...")
    start_time = time.time()
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)
    print(f"Optimized Model created in {time.time() - start_time:.2f} seconds.")

    # Evaluate all models on the validation set
    print("[+] Evaluating Models...")
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()


