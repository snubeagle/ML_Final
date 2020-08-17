import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error  # , make_scorer
from sklearn.model_selection import KFold       # , GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def vec_combine(v1, v2):
    result = []
    for x in v1:
        for y in v2:
            result.append((x, y))
    return result


class NN_Predictor:
    ''' Store a dataset, implement an MLPRegressor model with it, and
        record predictions. '''

    data = None
    model = None
    predictions = None
    folded_data = []

    def __init__(self, data=None):
        ''' Initialise model object by loading, cropping,
            and splitting the specified dataset '''
        if not data:
            raise Exception("A data set must be supplied to create"
                            + "a Predictor.")
        # print(data)
        if data[-4:] not in [".csv", ".txt"]:
            data = f"{data}.csv"
        filename = f"/{data}"
        filepath = os.getcwd() + filename
        self.data = pd.read_csv(filepath)
        self.crop_and_regularize()
        self.split_data()

    def split_data(self, numfolds=5):
        ''' Divide dataset into folds for testing & cross-validation '''
        self.kfolds = KFold(n_splits=numfolds, shuffle=True)
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1:]
        for train_inds, test_inds in self.kfolds.split(self.X):
            this_fold = {"train_X": self.X.iloc[train_inds, :],
                         "train_y": self.y.iloc[train_inds, :],
                         "test_X": self.X.iloc[test_inds, :],
                         "test_y": self.y.iloc[test_inds, :]}
            self.folded_data.append(this_fold)
        # print(self.folded_data[0]["train_X"])

    def crop_and_regularize(self):
        ''' Exclude data irrelevant to this operation; also create
            standardized versions of the data I guess?  Not using them yet,
            TBD if this is a good idea. '''
        drop_cols = ["flagCa", "flagMg", "flagK", "flagNa", "flagNH4",
                     "flagNO3", "flagCl", "flagSO4", "flagBr", "valcode",
                     "invalcode"]
        self.data.drop(columns=drop_cols, inplace=True)
        self.data = self.data.applymap(lambda x: np.NaN if x == -9 else x)
        self.data = self.data.loc[:, ['NO3', 'SO4', 'Cl', 'NH4']]
        self.data = self.data.query('NO3 != "NaN" & SO4 != "NaN"'
                                    + '& Cl != "NaN" & NH4 != "NaN"')
        scaler = StandardScaler()
        scaler.fit(np.array(self.data))
        self.std_data = pd.DataFrame(scaler.transform(self.data))

    def build(self):
        ''' Construct the NN model '''
        # mlpParams = {
        #     "hidden_layer_sizes": vec_combine(
        #         [n for n in range(10, 101) if n % 10 == 0],
        #         [n for n in range(10, 101) if n % 10 == 0]
        #     ),
        #     "activation": ["logistic", "tanh", "relu"],
        #     "solver": ["sgd", "adam"],
        #     "alpha": [0.005, 0.001, 0.0001]
        # }
        self.model = MLPRegressor(hidden_layer_sizes=(15, 12, 9, 6, 3),
                                  activation="relu",
                                  solver='adam',
                                  alpha=0.001,
                                  learning_rate='adaptive',
                                  max_iter=10000,
                                  shuffle=True)
        # reg = GridSearchCV(estimator=mlp,
        #                    param_grid=mlpParams,
        #                    n_jobs=-1,
        #                    scoring=make_scorer(mean_squared_error))
        # reg.fit(X=self.X, y=np.ravel(self.y))
        # print(reg.best_params_)
        # self.model.fit(self.X, np.ravel(self.y))
        # print("Model complete.")
        # print(self.model.coefs_)
        # self.predictions = self.model.predict(self.X)
        # print("MSE =", mean_squared_error(self.y, self.predictions))
        # print("R^2 =", r2_score(self.y, self.predictions))

    def plotz(self):
        ''' Make and plot predictions '''
        total_error = 0
        rem_plts = [231, 232, 233, 234, 235, 236]
        for fold in self.folded_data:
            self.model.fit(fold["train_X"], np.ravel(fold["train_y"]))
            pred = self.model.predict(fold["test_X"])
            plt.subplot(rem_plts[0])
            plt.plot(fold["test_X"], fold["test_y"], 'bs', alpha=0.3)
            plt.plot(fold["test_X"], pred, 'go', alpha=0.3)
            rem_plts = rem_plts[1:]
            mse = mean_squared_error(fold["test_y"], pred)
            plt.text(0, 0, s=f"MSE = {mse}")
            total_error += mse
        # plt.subplot(233)
        # plt.plot(self.X, self.y, 'ro', alpha = 0.25)
        # plt.plot(self.X, avg_preds, 'g^', alpha = 0.5)
        plt.show()
        print("Average MSE:", total_error / 5)


if __name__ == "__main__":
    print("Duh, I'm a brain machine.")
    unit = NN_Predictor("NTN-NE15-w.csv")
    unit.build()
    unit.plotz()
