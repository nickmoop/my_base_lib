from base_lib.helper import make_directory

CSV_DIRECTORY = 'csvs'
make_directory(CSV_DIRECTORY)


def replace_values_in_column_by_regex(
        dataframe, column_name, target_value, updated_value
):
    dataframe[column_name] = dataframe[column_name].replace(
        target_value, updated_value, regex=True)


def replace_values_in_column(dataframe, column_name, replacing_map):
    dataframe[column_name] = dataframe[column_name].map(replacing_map)


def replace_values(dataframe, target_values, updated_values):
    return dataframe.replace(target_values, updated_values)


def make_description_csv(dataframe, dataset_name):
    dataframe.describe().to_csv(
        '{}/{}_description.csv'.format(CSV_DIRECTORY, dataset_name))


def make_correlation_csv(dataframe, dataset_name):
    dataframe.corr().to_csv(
        '{}/{}_correlation_matrix.csv'.format(CSV_DIRECTORY, dataset_name))
