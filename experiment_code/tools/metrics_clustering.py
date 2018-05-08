from sklearn.metrics import homogeneity_score, adjusted_rand_score, adjusted_mutual_info_score, completeness_score, v_measure_score, fowlkes_mallows_score, silhouette_score, calinski_harabaz_score


def evaluate_model(m, X_train, y_train_pred, y_train, X_test, y_test, unique_labels, plot=False):
    truth_label_metrics = [homogeneity_score, adjusted_rand_score, adjusted_mutual_info_score, completeness_score, v_measure_score, fowlkes_mallows_score]
    non_truth_label_metrics = [silhouette_score, calinski_harabaz_score]

    cluster_labels = m.labels_
    results = {'labels': unique_labels, 'cluster_labels': cluster_labels,
               'train_labels': y_train}

    for metric in truth_label_metrics:
        metric_name = metric.__name__
        metric_score = metric(y_train, y_train_pred)
        results[metric_name] = metric_score

    if len(list(set(cluster_labels))) < 2:
        print('Just one cluster found, unsupervised cluster metrics not calculated')
    else:
        for metric in non_truth_label_metrics:
            metric_name = metric.__name__
            metric_score = metric(X_train, cluster_labels)
            results[metric_name] = metric_score

    if hasattr(m, 'predict'):
        y_test_pred = m.predict(X_test)
        for metric in truth_label_metrics:
            metric_name = '{}_{}'.format(metric.__name__, 'test')
            metric_score = metric(y_test, y_test_pred)
            results[metric_name] = metric_score

    return results
