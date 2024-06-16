import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(file_path):
    # Veri setini yükle
    data = pd.read_csv(file_path)

    # Hedef değişkeni ayrıştır
    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Özellikleri ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def prepare_data_for_cluster(file_path):
    # Veri setini yükle
    data = pd.read_csv(file_path)

    # Hedef değişkeni ayrıştır
    X = data.drop("Cluster", axis=1)
    y = data["Cluster"]

    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Özellikleri ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
