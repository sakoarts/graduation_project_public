import datetime
import pandas as pd

from keras.callbacks import Callback

class EvaluateModel(Callback):
    def __init__(self):
        super(Callback, self).__init__()


class MetricsToCSV(Callback):
    def __init__(self, filename):
        self.filename = filename
        self.header = True

    def on_epoch_end(self, epoch, logs={}):
        logs_ = dict(logs)
        logs_['epoch'] = epoch
        logs_['timestamp'] = datetime.datetime.now()
        metrics = [logs_]
        df = pd.DataFrame(metrics).set_index('epoch')
        df.to_csv(self.filename, mode='a', header=self.header)
        self.header = False
