import numpy as np
from spdt import SPDT, SPDTClassifierParallel
from sklearn.datasets import fetch_openml, load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def cv_10fold_percent_error(X, y, class_names, *, B=50, W=4, max_depth=15, min_samples_leaf=2, impurity="entropy"):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs_SPDT, perrs_SPDT = [], []
    accs_Base, perrs_Base = [], []
    
    Y_test_label = np.array([])
    SPDT_Y_pred = np.array([])
    Base_Y_pred = np.array([])
    

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        Y_test_label = np.append(Y_test_label, y_te)

        clf = SPDTClassifierParallel(B=B, W=W, max_depth=max_depth, min_samples_leaf=min_samples_leaf, impurity=impurity)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        perr = 100 * (1 - acc)
        accs_SPDT.append(acc)
        perrs_SPDT.append(perr)
        SPDT_Y_pred = np.append(SPDT_Y_pred, y_pred)
        
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        perr = 100 * (1 - acc)
        accs_Base.append(acc)
        perrs_Base.append(perr)
        Base_Y_pred = np.append(Base_Y_pred, y_pred)

    print(f"SPDT Classifier:")
    print(f"\tAccuracy: {np.mean(accs_SPDT):.4f} ± {np.std(accs_SPDT):.4f}")
    print(f"\tPercent error: {np.mean(perrs_SPDT):.2f}% ± {np.std(perrs_SPDT):.2f}%")
    print(f"Sklearn Decision Tree Classifier:")
    print(f"\tAccuracy: {np.mean(accs_Base):.4f} ± {np.std(accs_Base):.4f}")
    print(f"\tPercent error: {np.mean(perrs_Base):.2f}% ± {np.std(perrs_Base):.2f}%")
    
    # Plot Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    cm = confusion_matrix(Y_test_label, SPDT_Y_pred)
    sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names),
            annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    axes[0].set_title(f"Confusion Matrix of SPDT on UCI adult. B={B} W={W} tree_depth={max_depth}")
    
    cm = confusion_matrix(Y_test_label, Base_Y_pred)
    sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names),
            annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    axes[1].set_title(f"Confusion Matrix of Sklearn DT on UCI adult. B={B} W={W} tree_depth={max_depth}")
    plt.show()
    
    return np.array(accs_SPDT), np.array(perrs_SPDT)

if __name__ == "__main__":
    # rng = np.random.default_rng(7)
    # # Synthetic 2‑class, 6‑feature data where only first 2 features matter
    # n_train = 20000
    # n_test = 4000
    # d = 6
    # X = rng.normal(size=(n_train + n_test, d)).astype(float)
    # # non‑linear decision boundary on f0 and f1
    # y = ((X[:, 0] + 0.5 * X[:, 1] + 0.2 * (X[:, 0] ** 2) > 0.4)).astype(int)
    
    # Load UCI Adult (Census Income)
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X = adult.data.select_dtypes(include=[np.number]).to_numpy()  # keep numeric features
    y = (adult.target == ">50K").astype(int).to_numpy()           # binary label
    
    # Load Iris Data set
    # iris = load_iris()
    # X, y = iris.data, iris.target
    
    cv_10fold_percent_error(X, y, class_names=np.unique_values(y))
    
    # X_train, X_test = X[:n_train], X[n_train:]
    # Y_train, Y_test = y[:n_train], y[n_train:]
    # clf = SPDTClassifier(B=50, max_depth=8, min_samples_leaf=20, impurity="gini")
    # clf.fit(X_train, Y_train)
    # y_pred = clf.predict(X_test)
    # accuracy = np.mean(y_pred == Y_test)
    # print("Test accuracy:", accuracy)
    # percent_error = 100 * (1 - np.mean(y_pred == Y_test))
    # print(f"Percent error: {percent_error:.2f}%")


    # phat = clf.predict_proba(X_test)[:, 1]
    # yhat = (phat >= 0.5).astype(int)
    # acc = (yhat == Y_test).mean()
    # print(f"Test accuracy: {acc:.3f} | Tree nodes: {len(clf.tree.nodes)}")