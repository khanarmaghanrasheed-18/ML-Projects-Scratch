import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time

data = load_breast_cancer()

X = data.data       
y = data.target  # 0 = malignant, 1 = benign
y = np.where(y == 0, -1, 1)

np.random.seed(42)

feature_names = data.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

class StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1

    def transform(self, X):
        return (X - self.mean)/ self.std
    

class PCA:
    def __init__(self, final):
        self.final_dimensions = final

    def mean(self, X, d):
        total = 0
        for i in range(len(X)):
            total += X[i][d]
        return total / len(X)

    def covariance(self, X, d1, d2, feature_means):
        total = 0
        for i in range(len(X)):
            total += (X[i][d1] - feature_means[d1]) * (X[i][d2] - feature_means[d2])

        return total / (len(X) - 1)

    def variance(self, X, d, feature_means):
        total = 0
        for i in range(len(X)):
            total += (X[i][d] - feature_means[d]) ** 2

        return total / (len(X) - 1)

    def covariance_matrix(self, X, dimensions):
        feature_means = []
        for d in range(dimensions):
            feature_means.append(self.mean(X, d))

        cov_X = [[0 for _ in range(dimensions)] for _ in range(dimensions)]
        for i in range(dimensions):
            for j in range(dimensions):
                if i == j:
                    cov_X[i][j] = self.variance(X, i, feature_means)
                else:
                    cov_X[i][j] = self.covariance(X, i, j, feature_means)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_X) # extracting eigenvalues and eigenvectors

        return eigenvalues, eigenvectors

    def fit(self, X):
        self.feature_mean = np.mean(X, axis=0)
        X_centered = X - self.feature_mean
        dimensions = X.shape[1]

        self.eigenvalues, self.eigenvectors = self.covariance_matrix(X_centered, dimensions)
        sorted_indices = np.argsort(self.eigenvalues)[::-1]

        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]

        self.W = self.eigenvectors[:, :self.final_dimensions]

    def transform(self, X):
        X_centered = X - self.feature_mean
        return np.dot(X_centered, self.W)
    

class SVM:
    def fit(self, X, y, lr=0.05, epochs=1000, lambda_param = 0.05):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for epoch in range(epochs):
            for i in range(len(X)):
                condition = y[i] * (np.dot(X[i], self.w) + self.b)    
                if condition >= 1:
                    dw = 2 * lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * lambda_param * self.w - y[i] * X[i]
                    db = -y[i]

                self.w = self.w - lr * dw
                self.b = self.b - lr * db

    def predict(self, X):
        scores = np.dot(X, self.w) + self.b
        return np.where(scores >= 0, 1, -1)
            
class Pipeline:
    def __init__(self, std, pca, svm):
        self.std = std
        self.pca = pca
        self.svm = svm

    def fit(self, X, y):
        self.std.fit(X)
        X_scaled = self.std.transform(X)

        self.pca.fit(X_scaled)
        X_pca = self.pca.transform(X_scaled)

        self.svm.fit(X_pca, y)

    def predict(self, X):
        X_scaled = self.std.transform(X)
        X_pca = self.pca.transform(X_scaled)

        return self.svm.predict(X_pca)
    
    def visualize(self, X, y):
        X_scaled = self.std.transform(X)
        X_pca = self.pca.transform(X_scaled)

        w = self.svm.w
        b = self.svm.b

        margin_values = y * (np.dot(X_pca, w) + b)
        support_vector_indices = np.where(
            np.abs(margin_values - 1) <= 0.2
        )[0]

        support_vectors = X_pca[support_vector_indices]

        x_min = X_pca[:, 0].min() - 1
        x_max = X_pca[:, 0].max() + 1
        x_values = np.linspace(x_min, x_max, 200)

        decision_boundary = -(w[0] * x_values + b) / w[1]

        margin_positive = -(w[0] * x_values + b - 1) / w[1]
        margin_negative = -(w[0] * x_values + b + 1) / w[1]

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        ax[0].scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y, s=20, alpha=0.7
        )

        ax[0].set_title("Breast Cancer Dataset after PCA")
        ax[0].set_xlabel("Principal Component 1")
        ax[0].set_ylabel("Principal Component 2")
        ax[0].grid(True)

        ax[1].scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y, s=20, alpha=0.7
        )

        ax[1].plot(
            x_values,
            decision_boundary,
            linewidth=2,
            label="Decision Boundary"
        )

        ax[1].plot( x_values, margin_positive, linestyle="--", label="Positive Margin"
        )

        ax[1].plot(
            x_values, margin_negative, linestyle="--", label="Negative Margin"
        )

        if len(support_vectors) > 0:
            ax[1].scatter( support_vectors[:, 0], support_vectors[:, 1], s=160,
                facecolors="none", edgecolors="black", linewidths=2, label="Support Vectors"
            )

        ax[1].set_title("Linear SVM on PCA Features")
        ax[1].set_xlabel("Principal Component 1")
        ax[1].set_ylabel("Principal Component 2")

        ax[1].set_xlim(-5, 3)
        ax[1].set_ylim(-5, 7.5)

        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    def summary(self, X):
        X_scaled = self.std.transform(X)
        X_pca = self.pca.transform(X_scaled)

        print("\n========= STANDARD SCALER =========")
        print("Original Shape:", X.shape)
        print("Scaled Shape:", X_scaled.shape)

        print("\nFeature Means (first 5):")
        print(self.std.mean[:5])

        print("\nFeature Standard Deviations (first 5):")
        print(self.std.std[:5])
        time.sleep(2)

        print("\n========== PCA SUMMARY ==========")
        print("Original Shape:", X_scaled.shape)
        print("Projected Shape:", X_pca.shape)

        print("\nTop 2 Eigenvalues:")
        print(self.pca.eigenvalues[:2])

        print("\nPrincipal Component Matrix Shape:")
        print(self.pca.W.shape)
        time.sleep(2)

        print("\n=========== SVM SUMMARY ==========")
        print("Weights:", self.svm.w)
        print("Bias:", self.svm.b)
        print("Norm of w:", np.linalg.norm(self.svm.w))
        time.sleep(2)

pipeline = Pipeline(
    StandardScaler(),
    PCA(final=2),
    SVM()
)

pipeline.fit(X_train, y_train)
pipeline.summary(X_train)

predictions = pipeline.predict(X_test)
accuracy = np.mean(predictions == y_test) * 100

predictions = np.where(predictions == -1, "M", "B")

print("M = malignant, B = benign")
print("Predictions: \n", predictions)
print("Accuracy:", accuracy)
time.sleep(1)

pipeline.visualize(X_train, y_train)