from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(X_train, y_train):
    # CatBoost sınıflandırıcı modelini oluştur ve eğit
    model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='Logloss')
    model.fit(X_train, y_train, verbose=100)

    return model

def evaluate_model_performance(model, X_test, y_test, is_cluster=False):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if is_cluster:
        print(f'Model Accuracy for Cluster Prediction: {accuracy}')
    else:
        print(f'Model Accuracy with Normal Data: {accuracy}')

    # Diğer performans verileri
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
