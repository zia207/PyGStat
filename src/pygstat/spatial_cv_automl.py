"""K-fold cross-validation for H2O AutoML trend models (optional ``h2o`` extra)."""

import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

class KFoldCV_AutoML:
    def __init__(self, n_splits=5, max_runtime_secs=30, random_state=42):
        self.n_splits = n_splits
        self.max_runtime_secs = max_runtime_secs
        self.random_state = random_state

    def validate(self, rk_model, X, y, coords):
        X, y, coords = np.asarray(X), np.asarray(y), np.asarray(coords)
        n = len(y)
        predictions = np.full(n, np.nan)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        if h2o.connection() is None:
            h2o.init(max_mem_size="32G", nthreads=2)
        for train_idx, test_idx in kf.split(X):
            # Prepare data
            feature_names = [f"X{i}" for i in range(X.shape[1])]
            df_h2o = pd.DataFrame(X[train_idx], columns=feature_names)
            df_h2o['target'] = y[train_idx]
            train = h2o.H2OFrame(df_h2o)

            # AutoML
            aml = H2OAutoML(
                max_runtime_secs=self.max_runtime_secs,
                seed=self.random_state,
                exclude_algos=["DeepLearning"]  # speed up
            )
            aml.train(x=feature_names, y='target', training_frame=train)

            # Predict
            test_h2o = h2o.H2OFrame(pd.DataFrame(X[test_idx], columns=feature_names))
            pred = aml.leader.predict(test_h2o).as_data_frame(use_pandas=True).values.flatten()
            predictions[test_idx] = pred

        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        return {"rmse": rmse, "r2": r2, "predictions": predictions}

class LOOCV_AutoML:
    def validate(self, rk_model, X, y, coords):
        # LOO is too expensive for AutoML — raise error or use k=5
        raise NotImplementedError("LOO CV is computationally infeasible for AutoML. Use KFoldCV_AutoML instead.")