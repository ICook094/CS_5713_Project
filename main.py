import numpy as np
from spdt_main import spdt_main
from sklearn.datasets import fetch_openml, load_iris

if __name__ == "__main__":
    
    # Load UCI Adult (Census Income)
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X = adult.data.select_dtypes(include=[np.number]).to_numpy()  # keep numeric features
    y = (adult.target == ">50K").astype(int).to_numpy()           # binary label
    
    # Load Iris Data set
    # iris = load_iris()
    # X, y = iris.data, iris.target
    
    spdt_main(X, y)