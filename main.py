import preprocessing
import modeling

if __name__ == "__main__":
    # Normal veri setini hazırlayın
    X_train, X_test, y_train, y_test = preprocessing.prepare_data("C:/Users/emirh/Desktop/water_potability.csv")

    # Normal veri seti üzerinde modeli eğit ve değerlendir
    model_normal = modeling.train_model(X_train, y_train)
    print("Model with Normal Data:")
    modeling.evaluate_model_performance(model_normal, X_test, y_test)

    print("\n")

    # Kümeleme veri seti üzerinde Cluster sütununu tahmin eden modeli hazırlayın
    X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = preprocessing.prepare_data_for_cluster("C:/Users/emirh/Desktop/water_potability_with_clusters.csv")

    # Kümeleme veri seti üzerinde modeli eğit ve değerlendir
    model_cluster = modeling.train_model(X_train_cluster, y_train_cluster)
    print("Model to Predict Cluster:")
    modeling.evaluate_model_performance(model_cluster, X_test_cluster, y_test_cluster, is_cluster=True)
