import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from sklearn.model_selection import learning_curve

from base_lib.helper import make_directory

PLOTS_DIRECTORY = 'plots'
make_directory(PLOTS_DIRECTORY)


def make_sorted_bar_plots(dataframe, arguments_names_list, dataset_name):
    tmp_len = len(arguments_names_list)
    for figure_number, argument in enumerate(arguments_names_list):
        plt.subplot(tmp_len, 1, 1 + figure_number)
        make_sorted_bar_subplot(dataframe, argument)

    plt.savefig('{}/{}_bar.png'.format(PLOTS_DIRECTORY, dataset_name))


def make_scatter_plots(dataframe, arguments_pairs_names_list, dataset_name):
    tmp_len = len(arguments_pairs_names_list)
    for figure_number, arguments_pair in enumerate(arguments_pairs_names_list):
        plt.subplot(tmp_len, 1, 1 + figure_number)
        make_scatter_subplot(dataframe, arguments_pair[0], arguments_pair[1])

    plt.savefig('{}/{}_scatter.png'.format(PLOTS_DIRECTORY, dataset_name))


def make_sorted_bar_subplot(dataframe, x_name):
    tmp_dataframe = dataframe[x_name].value_counts().sort_index()
    x_values = tmp_dataframe.values
    y_values = tmp_dataframe.keys()

    plt.bar(y_values, x_values)
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=20)
    plt.xlabel(x_name)
    plt.ylabel('Count')
    plt.tight_layout()


def make_scatter_subplot(dataframe, x_name, y_name):
    x_values = dataframe[x_name].values
    y_values = dataframe[y_name].values

    plt.scatter(x_values, y_values)

    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()


def classification_data_visualization(
        dataframe, dataset_name,
        arguments_names_list, arguments_pairs_names_list
):
    plt.figure(1)
    plt.subplots(1, len(arguments_pairs_names_list), figsize=(10, 10))
    make_scatter_plots(dataframe, arguments_pairs_names_list, dataset_name)
    plt.close()

    plt.figure(1)
    plt.subplots(1, len(arguments_names_list), figsize=(10, 10))
    make_sorted_bar_plots(dataframe, arguments_names_list, dataset_name)
    plt.close()

    plt.figure(1)
    plt.subplots(1, 1, figsize=(10, 10))
    make_correlation_matrix_plot(dataframe, dataset_name)
    plt.close()


def make_fitting_curve(metrics_dict, fitting_name, dataset_name):
    tmp_len = len(metrics_dict)

    columns_count = 4
    rows_count = tmp_len // columns_count + 1

    plt.figure(1)
    plt.subplots(
        nrows=rows_count, ncols=columns_count,
        figsize=(4*columns_count, 4*rows_count)
    )

    for metric_number, metric_name in enumerate(metrics_dict, start=1):
        plt.subplot(rows_count, columns_count, metric_number)

        x_axis = list()
        y_axis = list()
        for x, y in metrics_dict[metric_name].items():
            x_axis.append(x)
            y_axis.append(y)

        plt.plot(x_axis, y_axis)

        plt.locator_params(axis='y', nbins=10)
        plt.locator_params(axis='x', nbins=10)
        plt.title(metric_name)
        plt.tight_layout()

    plt.savefig('{}/{}_{}_optimize.png'.format(
        PLOTS_DIRECTORY, dataset_name, fitting_name))
    plt.close()


def make_factor_subplot(dataframe, x_name, y_name):
    g = sns.factorplot(
        x=x_name, y=y_name, data=dataframe, kind='bar', size=6, palette='muted'
    )
    g.despine(left=True)


def make_factor_plots(dataframe, arguments_pairs_names_list, dataset_name):
    tmp_len = len(arguments_pairs_names_list)
    plt.figure(1)
    plt.subplots(1, 1, figsize=(4, 4 * tmp_len))
    for figure_number, arguments_pair in enumerate(arguments_pairs_names_list):
        plt.subplot(tmp_len, 1, 1 + figure_number)
        make_factor_subplot(dataframe, arguments_pair[0], arguments_pair[1])
        plt.savefig('{}/{}_{}_factor.png'.format(
            PLOTS_DIRECTORY, dataset_name, arguments_pair[0]))

    plt.close()


def make_correlation_matrix_plot(dataframe, dataset_name):
    plt.figure(1)
    tmp = dataframe.corr()
    plt.subplots(1, 1, figsize=(len(tmp) // 2, len(tmp) // 2))
    sns.heatmap(tmp, annot=True, fmt='.2f', cmap='coolwarm')
    plt.tight_layout()
    plt.savefig(
        '{}/{}_correlation_matrix.png'.format(PLOTS_DIRECTORY, dataset_name))
    plt.close()


def skewness_plot(dataframe, parameter_name):
    g = sns.distplot(
        dataframe[parameter_name], color='m',
        label='Skewness : {:.2}'.format(dataframe[parameter_name].skew())
    )
    g.legend(loc='best')
    g.set(ylabel=parameter_name)


def make_skewness_plots(dataframe, arguments_names_list, dataset_name):
    tmp_len = len(arguments_names_list)
    plt.figure(1)
    plt.subplots(1, 1, figsize=(4, 4*tmp_len))
    for figure_number, argument_name in enumerate(arguments_names_list):
        plt.subplot(tmp_len, 1, 1 + figure_number)
        skewness_plot(dataframe, argument_name)

    plt.tight_layout()
    plt.savefig('{}/{}_skewness.png'.format(PLOTS_DIRECTORY, dataset_name))
    plt.close()


def models_score_plot(cv_res, cv_std, dataset_name):
    plt.figure(1)
    # plt.subplots(1, 1, figsize=(6, 6))

    g = sns.barplot(
        'CrossValMeans', 'Algorithm',
        data=cv_res, palette='Set3', orient='h', **{'xerr': cv_std}
    )
    g.set_xlabel('Mean Accuracy')
    g.set_title('Cross validation scores')

    plt.tight_layout()
    plt.savefig(
        '{}/{}_models_scores.png'.format(PLOTS_DIRECTORY, dataset_name))
    plt.close()


def plot_learning_curve(
        estimator, title, x, y, ylim=None, cv=None, n_jobs=-1,
        train_sizes=numpy.linspace(.1, 1.0, 5)
):
    """Generate a simple plot of the test and training learning curve"""
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)

    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1, color='r'
    )
    plt.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color='g'
    )
    plt.plot(
        train_sizes, train_scores_mean, 'o-', color='r', label='Training score'
    )
    plt.plot(
        train_sizes, test_scores_mean, 'o-', color='g',
        label='Cross-validation score'
    )

    plt.legend(loc='best')


def make_learning_curves(
        dataset_name, x_train, y_train, estimators, kfold=5
):
    plt.figure(1)
    tmp_len = len(estimators)
    rows = tmp_len // 4 + 1
    plt.subplots(rows, 4, figsize=(10*tmp_len, 10))
    figure_number = 1
    for method_name, best_estimator in estimators.items():
        plt.subplot(rows, 4, figure_number)
        plot_learning_curve(
            best_estimator, '{} learning curves'.format(method_name),
            x_train, y_train, cv=kfold
        )
        figure_number += 1

    plt.tight_layout()
    plt.savefig('{}/{}_learning_curves.png'.format(
        PLOTS_DIRECTORY, dataset_name))
    plt.close()


def models_predictions_correlation_plot(predicted_values, dataset_name):
    plt.figure(1)
    sns.heatmap(predicted_values.corr(), annot=True)
    plt.tight_layout()
    plt.savefig('{}/{}_test_predicted.png'.format(
        PLOTS_DIRECTORY, dataset_name))
    plt.close()
