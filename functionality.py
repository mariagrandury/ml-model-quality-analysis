"""Functionality: groups analyses that evaluate how ”well“ an AI module performs a
given task (i.e. assessing the suitability of an AI module for an application domain). As
ML can be seen as a data-driven development technique we also categorize methods
that measure the quality of data into this pillar. Example analyses: Accuracy,
Confusion, Reliability, Fairness, Generalization"""

import tensorflow as tf

class FunctionalityAnalysis:

    def __init__(self, y_true, y_pred, task, nclass):
        self.y_true = y_true
        self.y_pred = y_pred
        self.task = task
        self.nclass = nclass

    def show_matrix_confusion(self):
        return tf.math.confusion_matrix(self.y_true, self.y_pred).numpy()

    def calculate_accuracy(self):
        acc = tf.keras.metrics.Accuracy()
        acc.update_state(self.y_true, self.y_pred)
        accuracy = acc.result().numpy()
        return accuracy

    # Check if balanced or imbalanced data

    def calculate_precision(self):
        precision = tf.keras.metrics.Precision()
        precision.update_state(self.y_true, self.y_pred)
        precision = precision.result().numpy()
        return precision

    def calculate_recall(self):
        recall = tf.keras.metrics.Recall()
        recall.update_state(self.y_true, self.y_pred)
        recall = recall.result().numpy()
        return recall

    def calculate_f1_score(self):
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score

    def evaluate_performance(self):
        if self.task == 'classification':
            self.show_matrix_confusion()
            print('Confusion Matrix: \n', self.show_matrix_confusion(), '\n')
            if self.nclass == 'binary':
                print('Accuracy:  ', self.calculate_accuracy(), '\n')
                print('Precision: ', self.calculate_precision())
                print('Recall:    ', self.calculate_recall())
                print('F1 Score:  ', round(self.calculate_f1_score(), 2))
            else:
                print('Accuracy:  ', self.calculate_accuracy(), '\n')
        else:
            print('Implement evaluation for regression problems!')
