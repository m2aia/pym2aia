from unittest import TestCase
import m2aia as m2
import numpy as np
import pathlib
import SimpleITK as sitk

def getTestData(relativePath:str)->str:
    return str(pathlib.Path(__file__).parent.joinpath(relativePath))

class TestImageIO(TestCase):


    def setUp(self):
        self.Image = m2.ImzMLReader(getTestData("data/test.imzML"))        
        self.eps = 1e-12
        self.tol_in_da = 5

    def test_IonImage_ExceptionThrownOnMzIsOutOfBounds(self):
        first,last = self.Image.GetXAxis()[[0,-1]]
        self.assertRaises(ValueError, lambda: self.Image.GetArray(first - self.eps, self.tol_in_da))
        self.assertRaises(ValueError, lambda: self.Image.GetArray(last + self.eps, self.tol_in_da) )


    def test_IonImage_ExceptionThrownOnMzIsOnBounds(self):
        first,last = self.Image.GetXAxis()[[0,-1]]
        # check if any value is not equal
        self.assertFalse(np.any(~np.equal(self.Image.GetArray(first, self.tol_in_da), np.load(getTestData("data/YS_LB_5.npy")))))
        self.assertFalse(np.any(~np.equal(self.Image.GetArray(last, self.tol_in_da), np.load(getTestData("data/YS_UB_5.npy")))))
        
    def test_sitk_IonImage_ExceptionThrownOnMzIsOnBounds(self):
        first,last = self.Image.GetXAxis()[[0,-1]]
        # check if any value is not equal
        self.assertFalse(np.any(~np.equal(sitk.GetArrayFromImage(self.Image.GetImage(first, self.tol_in_da)), np.load(getTestData("data/YS_LB_5.npy")))))
        self.assertFalse(np.any(~np.equal(sitk.GetArrayFromImage(self.Image.GetImage(last, self.tol_in_da)), np.load(getTestData("data/YS_UB_5.npy")))))

    def test_GetOrigin_ExceptionThrownOnMzIsOnBounds(self):
        origin = self.Image.GetOrigin()
        self.assertFalse(np.any(~np.equal(origin, [0,0,0])))
        print("Origin:", origin)

    def test_SetSmoothing(self):
        origin = self.Image.GetOrigin()
        self.assertFalse(np.any(~np.equal(origin, [0,0,0])))
        print("Origin:", origin)