import pandas
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
)

from base_lib.constants import (
    MODELS_METHODS, MODELS_ADDITIONAL_PARAMETERS, MODELS_HYPERPARAMETERS_GRID
)


class MyModelsEnsemble:
    def __init__(self, model_name, models_names):
        self.model_name = model_name
        self.models = list()
        self.voting_classifier = None
        self.generate_models(models_names)

    def generate_models(self, models_names):
        for model_name in models_names:
            print('Init model {}'.format(model_name))
            my_model = MyModel(model_name, 'mean')
            self.models.append(my_model)

    def get_best_hyper_parameters(self, x_train, y_train):
        for model in self.models:
            print('Finding hyperparameters {}'.format(model.name))
            model.get_best_hyper_parameters(x_train, y_train)

    def make_voting_classifier(self, voting='soft', n_jobs=2):
        print('Vote classifier, voting: {}, n_jobs: {}'.format(voting, n_jobs))
        self.voting_classifier = VotingClassifier(
            voting=voting, n_jobs=n_jobs,
            estimators=[(model.name, model.model) for model in self.models]
        )

    def score_models(self, x_train, y_train, kfold=10):
        cv_results = list()
        for model in self.models:
            cv_results.append(
                cross_val_score(
                    model.model, x_train, y=y_train,
                    scoring='accuracy', cv=kfold, n_jobs=2
                )
            )

        cv_means = list()
        cv_std = list()
        for cv_result in cv_results:
            cv_means.append(cv_result.mean())
            cv_std.append(cv_result.std())

        cv_res = pandas.DataFrame({
            'CrossValMeans': cv_means,
            'CrossValerrors': cv_std,
            'Algorithm': [model.name for model in self.models]
        })

        cv_res = cv_res.sort_values('CrossValMeans', ascending=False)

        return cv_res, cv_std


class MyModel:
    def __init__(self, model_name, fold_name):
        self.name = model_name
        self.fold_name = fold_name
        self.model = MODELS_METHODS[model_name](
            **MODELS_ADDITIONAL_PARAMETERS[model_name])
        self.hyperparameters_grid = MODELS_HYPERPARAMETERS_GRID[model_name]
        self.best_hyper_parameters = dict()

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        self.model = self.model(**self.best_hyper_parameters).fit(x, y)

    def get_best_hyper_parameters(self, train_x, train_y):
        self.model = GridSearchCV(
            self.model, param_grid=self.hyperparameters_grid, cv=10,
            scoring='accuracy', n_jobs=2
        )
        self.model.fit(train_x, train_y)
        self.best_hyper_parameters = self.model.best_estimator_


class MyData:
    def __init__(
            self, dataframe, prediction_target, features_vectors_names,
            transform_method=None
    ):
        self.transform_method = transform_method
        self.dataframe = self.transform_dataframe(dataframe)
        self.prediction_target = prediction_target
        self.features_vectors_names = features_vectors_names
        self.x = {'total': {'total': self.get_features_vectors(dataframe)}}
        self.y = {'total': {'total': self.make_prediction_target()}}
        self.split_method = None
        self.make_folds_split_method()

    def __repr__(self):
        return '{} folds, {} total features shape'.format(
            len(self.x)-1, self.x['total']['total'].shape)

    def split_train_test(
            self, test_size=0.2, random_state=0, shuffle=True, stratify=None
    ):
        train_x, valid_x, train_y, valid_y = train_test_split(
            self.x['total']['total'], self.y['total']['total'],
            test_size=test_size,
            train_size=1-test_size,
            shuffle=shuffle,
            stratify=stratify,
            random_state=random_state
        )

        self.x['total']['train'] = train_x
        self.x['total']['valid'] = valid_x
        self.y['total']['train'] = train_y
        self.y['total']['valid'] = valid_y

    def split_to_folds(self):
        x = self.x['total']['train']
        y = self.y['total']['train']
        fold_number = 1
        for train_indexes, valid_indexes in self.split_method(x, y):
            x_train, x_valid = x.iloc[train_indexes], x.iloc[valid_indexes]
            y_train, y_valid = y.iloc[train_indexes], y.iloc[valid_indexes]
            self.x[fold_number] = {'train': x_train, 'valid': x_valid}
            self.y[fold_number] = {'train': y_train, 'valid': y_valid}
            fold_number += 1

    def transform_dataframe(self, dataframe):
        if self.transform_method:
            return self.transform_method(dataframe)

        return dataframe

    def make_folds_split_method(
            self, n_splits=3, shuffle=True, random_state=1
    ):
        self.split_method = StratifiedKFold(
            shuffle=shuffle, n_splits=n_splits, random_state=random_state
        ).split

    def add_test_dataframe(self, dataframe):
        dataframe = self.transform_dataframe(dataframe)
        self.x['total']['test'] = self.get_features_vectors(dataframe)

    def make_prediction_target(self):
        return getattr(self.dataframe, self.prediction_target, None)

    def get_features_vectors(self, dataframe):
        return dataframe[self.features_vectors_names]
