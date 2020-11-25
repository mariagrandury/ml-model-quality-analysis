import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics \
    import confusion_matrix, roc_curve, classification_report, \
    max_error, mean_absolute_error, mean_squared_error, r2_score


class QualityAnalysis:
    """Perform quality analyses of an ML model.

    Attributes:
    -----------
    y_true: expected labels
    y_pred: predicted labels
    task: machine learning task

    Methods:
    --------
    plot_confusion_matrix(): plot confusion matrix
    plot_roc(): plot ROC curve
    calculate_accuracy:
    """

    def __init__(self, y_true, y_pred, task):
        self.y_true = y_true
        self.y_pred = y_pred
        self.task = task


    def plot_confusion_matrix(self, p=0.5):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion Matrix @{:.2f}'.format(p))
        plt.ylabel('Expected label')
        plt.xlabel('Predicted label')
        plt.show()


    def plot_roc(self, **kwargs):
        """Plot Receiver Operating Characteristic (ROC) curve."""
        fp, tp, _ = sklearn.metrics.roc_curve(self.y_true, self.y_pred)
        plt.figure(figsize=(5, 5))
        plt.plot(100 * fp, 100 * tp,linewidth=2, **kwargs)
        plt.title('ROC Curve')
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.xlim([-0.5, 20])
        plt.ylim([80, 100.5])
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()


    def classification_report(self):
        """Calculate the main classification metrics."""
        print(classification_report(self.y_true, self.y_pred))


    def regression_report(self):
        """Calculate the main regression metrics."""
        me = max_error(self.y_true, self.y_pred)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        mse = mean_squared_error(self.y_true, self.y_pred)
        r2 = r2_score(self.y_true, self.y_pred)
        print('max error                       ', me)
        print('mean absolute error             ', mae)
        print('mean squared error              ', mse)
        print('coefficient of determination    ', r2)


    def plot_scatter(self):
        """Plot scatter plot of expected vs predicted labels."""
        plt.figure(figsize = (5, 5))
        plt.scatter(self.y_true, self.y_pred)
        plt.ylabel('Expected label')
        plt.xlabel('Predicted label')


    def evaluate(self):
        """Evaluate ML model performance comparing expected and predicted labels."""
        if 'classification' in self.task:
            self.classification_report()
            self.plot_confusion_matrix()
            if 'binary' in self.task:
                self.plot_roc()
        else:
            self.regression_report()
            self.plot_scatter()
