from rich.progress import Progress
from rich.console import Console

import numpy as np
import pandas as pd

'''
import tensorflow as tf

class KNN(tf.keras.Model):
    def __init__(self, k):
        super(KNN, self).__init__()
        self.k = k

    def fit(self, X_train, y_train, X_test, y_test):
        self.X_train = tf.constant(X_train, dtype=tf.double)
        self.y_train = tf.constant(y_train, dtype=tf.int32)
        self.X_test = tf.constant(X_test, dtype=tf.double)
        self.y_test = tf.constant(y_test, dtype=tf.int32)

    def predict(self, show):
        predictions = []
        if show == True:
            with Progress() as progress:
                # Add a progress task
                task = progress.add_task("  progress:", total=len(self.X_test))

                for test_point in self.X_test:
                    # Compute Euclidean distance from the test point to all training points
                    distances = tf.norm(self.X_train - test_point, axis=1)
                    # Find the indices of the k nearest neighbors
                    k_indices = tf.argsort(distances)[:self.k]
                    # Get the labels of the k nearest neighbors
                    k_labels = tf.gather(self.y_train, k_indices)
                    # Predict the majority class (mode)
                    unique_labels, _, counts = tf.unique_with_counts(k_labels)
                    majority_class = unique_labels[tf.argmax(counts)]
                    predictions.append(int(majority_class))
                    
                    #if (len(predictions) % 10 == 0):
                    #    print(f"[INFO] {len(predictions)}/{len(X_test)}")

                    progress.update(task, advance=1)

        elif show == False:
            for test_point in self.X_test:
                # Compute Euclidean distance from the test point to all training points
                distances = tf.norm(self.X_train - test_point, axis=1)
                # Find the indices of the k nearest neighbors
                k_indices = tf.argsort(distances)[:self.k]
                # Get the labels of the k nearest neighbors
                k_labels = tf.gather(self.y_train, k_indices)
                # Predict the majority class (mode)
                unique_labels, _, counts = tf.unique_with_counts(k_labels)
                majority_class = unique_labels[tf.argmax(counts)]
                predictions.append(int(majority_class))
                if (len(predictions) % 10 == 0):
                    print(f"[INFO] {len(predictions)}/{len(self.X_test)}")
        return predictions

    def save_model(self, filepath):
        np.savez(filepath, X_train=self.X_train.numpy(), y_train=self.y_train.numpy(), X_test=self.X_test.numpy(), y_test=self.y_test.numpy())

    def load_model(self, filepath):
        data = np.load(filepath)
        self.X_train = tf.constant(data['X_train'], dtype=tf.double)
        self.y_train = tf.constant(data['y_train'], dtype=tf.int32)
        self.X_test = tf.constant(data['X_train'], dtype=tf.double)
        self.y_test = tf.constant(data['y_train'], dtype=tf.int32)
'''