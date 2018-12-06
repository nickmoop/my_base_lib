from base_lib.metrics_calculations import calculate_classification_metrics


def TMP_calculate_hyper_parameters(TMP_model_trainer, train_x, train_y, val_x, val_y, TMP_interval, hyper_parameter_name):
    fitting = dict()
    for max_leaf_count in TMP_interval:
        arguments = {hyper_parameter_name: max_leaf_count}
        decision_tree_model = TMP_model_trainer(
            random_state=1, **arguments)
        decision_tree_model.fit(train_x, train_y)

        predicted_values = decision_tree_model.predict(val_x)
        metrics = calculate_classification_metrics(val_y, predicted_values)
        for metric_name in metrics:
            if metric_name not in fitting:
                fitting[metric_name] = dict()

            fitting[metric_name][max_leaf_count] = metrics[metric_name]

    return fitting


def get_optimal_hyper_parameters(fitting):
    best_parameters = dict()
    for metric_name, metric_data in fitting.items():
        for hyper_parameter_value, metric_value in metric_data.items():
            if metric_name not in best_parameters:
                best_parameters[metric_name] = {
                    'metric_value': metric_value,
                    'hyperparameter_value': hyper_parameter_value
                }
            elif metric_value > best_parameters[metric_name]['metric_value']:
                best_parameters[metric_name]['metric_value'] = metric_value
                best_parameters[metric_name]['hyperparameter_value'] = hyper_parameter_value

    return best_parameters
