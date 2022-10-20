import tensorflow as tf
from .. import BatchGenerator
from .. import Dataset

class BatchSequence(tf.keras.utils.Sequence):
    def __init__(self, dataset: Dataset.BaseDataSet , batch_size: int, shuffle: bool=True):
        super().__init__()
        self.gen = BatchGenerator(dataset, batch_size, shuffle)
    
    def __len__(self):
        return self.gen.__len__()

    def on_epoch_end(self):
        self.gen.on_epoch_end()
    
    def __getitem__(self, index):
        return self.gen.__getitem__(index)
