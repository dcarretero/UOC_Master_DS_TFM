import pandas as pd
import numpy as np

class AnalyticsHelper:
    def getBinaryClassificationConfusionMatrixMetrics(self, confusionMatrix):
        # rows = gt, col = pred
        metrics={}
        tp = confusionMatrix[1,1]
        tn = confusionMatrix[0,0]
        fn = confusionMatrix[1,0]
        fp = confusionMatrix[0,1]

        metrics['recall'] = tp / (tp + fn)
        metrics['precision'] = tp / (tp + fp)
        metrics['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
        metrics['specificity'] = tn / (tn + fp)
        metrics['f1_score'] = 2 * ((metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']))

        return metrics
