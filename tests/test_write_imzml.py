import m2aia as m2

from unittest import TestCase
import pathlib
import numpy as np
import random

def getTestData(relativePath:str)->str:
    return str(pathlib.Path(__file__).parent.joinpath(relativePath))

class TestWriteImzML(TestCase):

    def setUp(self):
        random.seed(42)
        self.Image = m2.ImzMLReader(getTestData("data/test.imzML"))
        self.Image.SetBaselineCorrection(m2.m2BaselineCorrectionTopHat, 50)

    def test_WriteImzML_ContinuousCentroids(self):
        self.Image.SetTolerance(50)
        self.Image.WriteContinuousCentroidImzML("/tmp/continuous_centroid.imzML", [x for x in range(2000,3000,20)])

        ImageTmp = m2.ImzMLReader("/tmp/continuous_centroid.imzML")        
        ImageRef = m2.ImzMLReader(getTestData("data/continuous_centroid.imzML"))


        self.assertFalse(np.any(~np.equal(ImageRef.GetXAxis(), ImageTmp.GetXAxis())))

        for a,b in zip(ImageTmp.SpectrumIterator(),ImageRef.SpectrumIterator()):
                self.assertFalse(np.any(~np.equal(a[1], b[1])))
                self.assertFalse(np.any(~np.equal(a[2], b[2])))
        
        



