import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tmp_base_lib.helper import make_directory

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


def make_correlation_matrix_plot(dataframe, dataset_name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    tmp = dataframe.corr()
    cax = ax1.imshow(tmp, interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
    ax1.grid(True)
    plt.title(dataset_name)
    labels = ['empty_label'] + tmp.index.tolist()
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    fig.colorbar(cax, ticks=numpy.arange(-1, 1.25, 0.25))
    plt.savefig(
        '{}/{}_correlation_matrix.png'.format(PLOTS_DIRECTORY, dataset_name))


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
