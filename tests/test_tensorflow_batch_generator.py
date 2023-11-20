import m2aia as m2
import tensorflow as tf
from unittest import TestCase
import pathlib
import numpy as np
import random

def getTestData(relativePath:str)->str:
    return str(pathlib.Path(__file__).parent.joinpath(relativePath))

class BatchSequence(tf.keras.utils.Sequence):
    def __init__(self, dataset: m2.Dataset.BaseDataSet , batch_size: int, shuffle: bool=True):
        super().__init__()
        self.gen = m2.BatchGenerator(dataset, batch_size, shuffle)
    
    def __len__(self):
        return self.gen.__len__()

    def on_epoch_end(self):
        self.gen.on_epoch_end()
    
    def __getitem__(self, index):
        return self.gen.__getitem__(index)
    

class TestTensorflowBatchGenerator(TestCase):


    def setUp(self):
        random.seed(42)
        self.Image = m2.ImzMLReader(getTestData("data/test.imzML"))
        self.eps = 1e-12
        self.epochs = 1
        self.dataset = m2.SpectrumDataset([self.Image])
        self.sequence = BatchSequence(self.dataset, batch_size=10)
       

    def test_IonImage_ExceptionThrownOnMzIsOutOfBounds(self):
        
      pass




