from typing import Literal
from ctypes import create_string_buffer, c_void_p, c_uint32, c_char_p, c_double, c_float, c_ushort , POINTER
import pathlib
import numpy as np
import SimpleITK as sitk

from .Library import get_library



m2NormalizationTIC: str = 'TIC'
m2NormalizationSum: str = 'Sum'
m2NormalizationMean: str = 'Mean'
m2NormalizationMax: str = 'Max'
m2NormalizationMedian: str = 'Median'
m2NormalizationInFile: str = 'InFile'
m2NormalizationRMS: str = 'RMS'
m2Normalization = Literal[f"{m2NormalizationTIC}", f"{m2NormalizationSum}", f"{m2NormalizationMean}",
                          f"{m2NormalizationMax}", f"{m2NormalizationMedian}", f"{m2NormalizationInFile}", 
                          f"{m2NormalizationRMS}", "None"]

m2SmoothingSavitzkyGolay: str = "SavitzkyGolay"
m2SmoothingGaussian: str = "Gaussian"
m2Smoothing = Literal[f"{m2SmoothingSavitzkyGolay}", f"{m2SmoothingGaussian}", "None"]

m2PoolingMean: str = "Mean"
m2PoolingMedian: str = "Median"
m2PoolingMaximum: str = "Maximum"
m2PoolingSum: str = "Sum"
m2Pooling = Literal[f"{m2PoolingMean}", f"{m2PoolingMedian}",
                    f"{m2PoolingMaximum}", f"{m2PoolingSum}"]

m2BaselineCorrectionTopHat: str = "TopHat"
m2BaselineCorrectionMedian: str = "Median"
m2BaselineCorrection = Literal[f"{m2BaselineCorrectionTopHat}",
                               f"{m2BaselineCorrectionMedian}", "None"]

m2IntensityTransformationLog2: str = "Log2"
m2IntensityTransformationLog10: str = "Log10"
m2IntensityTransformationSquareRoot: str = "SquareRoot"
m2IntensityTransformation = Literal[f"{m2IntensityTransformationLog2}",
                                    f"{m2IntensityTransformationLog10}", f"{m2IntensityTransformationSquareRoot}", "None"]


class ImzMLReader(object):
    def __init__(self, imzML_path, baseline_correction: m2BaselineCorrection = "None",
                 baseline_correction_half_window_size: int = 50,
                 normalization: m2Normalization = "None",
                 smoothing: m2Smoothing = "None",
                 smoothing_half_window_size: int = 2,
                 intensity_transformation: m2IntensityTransformation = "None",
                 pooling: m2Pooling = m2PoolingMaximum):

        self.baseline_correction = baseline_correction
        self.baseline_correction_hws = baseline_correction_half_window_size
        self.intensity_transformation = intensity_transformation
        self.normalization = normalization
        self.pooling = pooling
        self.smoothing = smoothing
        self.smoothing_hws = smoothing_half_window_size

        self.lib = get_library()

        HANDLE_PTR = c_void_p

        self.lib.CreateImageHandle.argtypes = [
            c_char_p, c_char_p]
        self.lib.CreateImageHandle.restype = HANDLE_PTR

        self.lib.DestroyImageHandle.argtypes = [HANDLE_PTR]
        self.lib.DestroyImageHandle.restype = None

        self.lib.GetSize.argtypes = [HANDLE_PTR, POINTER(c_uint32)]
        self.lib.GetSize.restype = None

        self.lib.GetSpacing.argtypes = [
            HANDLE_PTR, POINTER(c_double)]
        self.lib.GetSpacing.restype = None

        self.lib.GetXAxis.argtypes = [
            HANDLE_PTR, POINTER(c_double)]
        self.lib.GetXAxis.restype = None

        self.lib.GetXAxisDepth.argtypes = [HANDLE_PTR]
        self.lib.GetXAxisDepth.restype = c_uint32

        self.lib.GetImageArrayFloat64.argtypes = [
            HANDLE_PTR, c_double, c_double, POINTER(c_double)]
        self.lib.GetImageArrayFloat64.restype = None

        self.lib.GetImageArrayFloat32.argtypes = [
            HANDLE_PTR, c_double, c_double, POINTER(c_float)]
        self.lib.GetImageArrayFloat32.restype = None

        self.lib.GetMaskArray.argtypes = [
            HANDLE_PTR, POINTER(c_ushort)]
        self.lib.GetMaskArray.restype = None

        self.lib.GetIndexArray.argtypes = [
            HANDLE_PTR, POINTER(c_uint32)]
        self.lib.GetIndexArray.restype = None

        self.lib.GetSpectrumType.argtypes = [HANDLE_PTR]
        self.lib.GetSpectrumType.restype = c_uint32

        self.lib.GetSpectrumDepth.argtypes = [HANDLE_PTR, c_uint32]
        self.lib.GetSpectrumDepth.restype = c_uint32

        self.lib.GetSizeInBytesOfYAxisType.argtypes = [HANDLE_PTR]
        self.lib.GetSizeInBytesOfYAxisType.restype = c_uint32

        self.lib.GetMeanSpectrum.argtypes = [
            HANDLE_PTR, POINTER(c_double)]
        self.lib.GetMeanSpectrum.restype = None

        self.lib.GetSpectrumPosition.argtypes = [
            HANDLE_PTR, c_uint32, POINTER(c_uint32)]
        self.lib.GetSpectrumPosition.restype = None

        self.lib.GetMaxSpectrum.argtypes = [
            HANDLE_PTR, POINTER(c_double)]
        self.lib.GetMaxSpectrum.restype = None

        self.lib.GetYDataTypeSizeInBytes.argtypes = [HANDLE_PTR]
        self.lib.GetYDataTypeSizeInBytes.restype = c_uint32

        self.lib.GetNumberOfSpectra.argtypes = [HANDLE_PTR]
        self.lib.GetNumberOfSpectra.restype = c_uint32

        self.lib.GetMetaDataDictionary.argtypes = [HANDLE_PTR]
        self.lib.GetMetaDataDictionary.restype = c_char_p

        self.lib.DestroyCharBuffer.argtypes = [HANDLE_PTR]
        self.lib.DestroyCharBuffer.restype = None

        self.lib.GetSpectrum.argtypes = [HANDLE_PTR, c_uint32, POINTER(c_float), POINTER(c_float)]
        self.lib.GetSpectrum.restype = None

        self.lib.GetSpectra.argtypes = [HANDLE_PTR, c_void_p, c_uint32, POINTER(c_float)]
        self.lib.GetSpectra.restype = None

        self.lib.GetIntensities.argtypes = [HANDLE_PTR, c_void_p, c_uint32, POINTER(c_float)]
        self.lib.GetIntensities.restype = None

        self.x_axis = None
        
        self.imzML_path = imzML_path
        self.handle = None
        self.spectrum_type_id = None
        self.spectrum_types = {0: "None", 1: "ContinuousProfile",
                               2: "ProcessedProfile", 4: "ContinuousCentroid", 8: "ProcessedCentroid"}

    def __enter__(self):
        self.Execute()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.lib.DestroyImageHandle(self.handle)

    def __delete__(self):
        self.lib.DestroyImageHandle(self.handle)

    def path(self) -> pathlib.Path:
        '''Absolute path to the referenced imzML'''
        return pathlib.Path(self.imzML_path)

    def dir(self) -> pathlib.Path:
        '''Absolute path to directory containing the referenced imzML '''
        return self.path().parent

    def name(self) -> str:
        '''Name (including file ending) of the given imzML'''
        return self.path().name

    def Execute(self):
        cPath = create_string_buffer(self.imzML_path.encode())
        parameters = self.GetParametersAsFormattedString()
        cParamPath = create_string_buffer(parameters.encode())
        self.handle = self.lib.CreateImageHandle(cPath, cParamPath)
    
        self.number_of_spectra = self.lib.GetNumberOfSpectra(self.handle)
        self.shape = self.GetShape()

        self.depth = self.lib.GetXAxisDepth(self.handle)
        self.spectrum_type_id = self.lib.GetSpectrumType(self.handle)

        # XAxis
        self.x_axis = np.zeros(self.depth, dtype=np.float64)
        self.lib.GetXAxis(self.handle, self.x_axis.ctypes.data_as(
            POINTER(c_double)))
        
        # mean overview spectrum
        self.mean_spectrum = np.zeros(self.depth, dtype=np.float64)
        self.lib.GetMeanSpectrum(self.handle, self.mean_spectrum.ctypes.data_as(
            POINTER(c_double)))

        # max overview spectrum
        self.max_spectrum = np.zeros(self.depth, dtype=np.float64)
        self.lib.GetMaxSpectrum(self.handle, self.max_spectrum.ctypes.data_as(
            POINTER(c_double)))
        
        return self

    def CheckHandle(self):
        if self.handle is None:
            raise Exception("Please initialize image handle by providing a valid file name and run the Execute() function of the reader!")

    def GetParametersAsFormattedString(self):        
        s = str()
        s += f"(baseline-correction {self.baseline_correction})\n"
        s += f"(baseline-correction-hw {self.baseline_correction_hws})\n"
        s += f"(smoothing {self.smoothing})\n"
        s += f"(smoothing-hw {self.smoothing_hws})\n"
        s += f"(normalization {self.normalization})\n"
        s += f"(pooling {self.pooling})\n"
        s += f"(transform {self.intensity_transformation})\n"
        return s

    def SetSmoothing(self, strategy: m2Smoothing, half_window_size=2):
        self.smoothing = strategy
        self.smoothing_hws = half_window_size

    def SetBaselineCorrection(self, strategy: m2BaselineCorrection, half_window_size=50):
        self.baseline_correction = strategy
        self.baseline_correction_hws = half_window_size

    def SetNormalization(self, strategy: m2Normalization):
        self.normalization = strategy

    def SetPooling(self, strategy: m2Pooling):
        self.pooling = strategy

    def SetIntensityTransformation(self, strategy: m2IntensityTransformation):
        self.intensity_transformation = strategy

    def GetYDataType(self):
        self.CheckHandle()
        size_in_bytes = self.lib.GetYDataTypeSizeInBytes(self.handle)
        if size_in_bytes == 4:
            return np.float32
        elif size_in_bytes == 8:
            return np.float64
        else:
            return None

    def GetShape(self):
        self.CheckHandle()
        shape = np.zeros((3), dtype=np.int32)
        self.lib.GetSize(self.handle, shape.ctypes.data_as(
            POINTER(c_uint32)))
        return shape


    def GetSpacing(self):
        self.CheckHandle()
        spacing = np.zeros((3), dtype=np.float64)
        self.lib.GetSpacing(self.handle, spacing.ctypes.data_as(
            POINTER(c_double)))
        return spacing

    def GetSpectrumPosition(self, id):
        # return x, y, z
        self.CheckHandle()
        pos = np.zeros((3), dtype=np.int32)
        self.lib.GetSpectrumPosition(self.handle, id, pos.ctypes.data_as(
            POINTER(c_uint32)))
        return pos

    def GetMetaData(self):
        self.CheckHandle()
        data = self.lib.GetMetaDataDictionary(self.handle)
        return data.decode("utf-8").split('\n')

    def GetOrigin(self):
        self.CheckHandle()
        origin = np.zeros((3), dtype=np.float64)
        self.lib.GetOrigin(self.handle, origin.ctypes.data_as(
            POINTER(c_double)))
        return origin

    def GetMaskArray(self):
        self.CheckHandle()
        slice = np.zeros(self.GetShape()[::-1], dtype=np.ushort)
        self.lib.GetMaskArray(self.handle, slice.ctypes.data_as(POINTER(c_ushort)))
        return slice

    def GetMaskImage(self):
        self.CheckHandle()
        slice = self.GetMaskArray()
        spacing = self.GetSpacing()
        origin = self.GetOrigin()
        I = sitk.GetImageFromArray(slice)
        I.SetSpacing(spacing)
        I.SetOrigin(origin)
        return I

    def GetIndexArray(self):
        self.CheckHandle()
        slice = np.zeros(self.GetShape()[::-1], dtype=np.uint32)
        self.lib.GetIndexArray(self.handle, slice.ctypes.data_as(POINTER(c_uint32)))
        return slice

    def GetIndexImage(self):
        self.CheckHandle()
        slice = self.GetIndexArray()
        spacing = self.GetSpacing()
        origin = self.GetOrigin()
        I = sitk.GetImageFromArray(slice)
        I.SetSpacing(spacing)
        I.SetOrigin(origin)
        return I
    
    def GetArray(self, center, tol, dtype=np.float32, squeeze:bool=False):
        self.CheckHandle()
        xs = self.GetXAxis()

        if center < np.min(xs) or center > np.max(xs):
            raise ValueError("Center is out of x-axis range!")

        slice = np.zeros(self.GetShape()[::-1], dtype=dtype)
        if dtype == np.float32:
            self.lib.GetImageArrayFloat32(
                self.handle, center, tol, slice.ctypes.data_as(POINTER(c_float)))
        elif dtype == np.float64:
            self.lib.GetImageArrayFloat64(
                self.handle, center, tol, slice.ctypes.data_as(POINTER(c_double)))
        else:
            raise TypeError(
                "Image pixel type is one of [np.float32, np.float64].")

        if squeeze:
            return np.squeeze(slice)
        return slice

    def GetMeanSpectrum(self) -> np.array:
        self.CheckHandle()
        return self.mean_spectrum

    def GetMaxSpectrum(self) -> np.array:
        self.CheckHandle()
        return self.max_spectrum

    def GetSkylineSpectrum(self) -> np.array:
        return self.GetMaxSpectrum()

    def GetXAxis(self) -> np.array:
        self.CheckHandle()
        return self.x_axis

    def GetXAxisDepth(self) -> int:
        self.CheckHandle()
        return self.depth

    def GetSpectrumDepth(self, id) -> int:
        self.CheckHandle()
        depth = self.lib.GetSpectrumDepth(self.handle, id)
        if depth <= 0:
            raise RuntimeError("Spectrum depth can not be 0!")
        return depth

    def GetSpectrumType(self):
        self.CheckHandle()
        return self.spectrum_types[self.spectrum_type_id]

    def GetSizeInBytesOfYAxisType(self):
        self.CheckHandle()
        return self.lib.GetSizeInBytesOfYAxisType(self.handle)

    def GetImage(self, mz, tol, dtype=np.float32):
        array = self.GetArray(mz, tol, dtype)
        spacing = self.GetSpacing()
        origin = self.GetOrigin()
        I = sitk.GetImageFromArray(array)
        I.SetSpacing(spacing)
        I.SetOrigin(origin)
        return I

    def GetSpectrum(self, index):
        self.CheckHandle()
        if index < 0 or index >= self.number_of_spectra:
            raise IndexError(
                "Index " + str(index) + " out of range of valid spectrum indices [0," + str(self.number_of_spectra - 1) + "] ")

        depth = self.GetSpectrumDepth(index)
        xs = np.zeros(depth, dtype=np.float32)
        ys = np.zeros(depth, dtype=np.float32)

        self.lib.GetSpectrum(self.handle, index, xs.ctypes.data_as(
            POINTER(c_float)), ys.ctypes.data_as(
            POINTER(c_float)))

        return [xs, ys]

    def GetIntensities(self, index):
        self.CheckHandle()
        if index < 0 or index >= self.number_of_spectra:
            raise IndexError(
                "Index " + str(index) + " out of range of valid spectrum indices [0," + str(self.number_of_spectra - 1) + "] ")

        if not 'Continuous' in self.GetSpectrumType():
            raise Exception(f"The ImzML format has to be in Continuous format! Current format ist {self.GetSpectrumType()}!\nUse GetSpectrum(...) instead.")

        ys = np.zeros(self.depth, dtype=np.float32)

        self.lib.GetIntensities(self.handle, index, ys.ctypes.data_as(POINTER(c_float)))
        return ys


    def GetSpectra(self, indices):
        self.CheckHandle()
        for index in indices:
            if index < 0 or index >= self.number_of_spectra:
                raise IndexError(
                    "Index " + str(index) + " out of range of valid spectrum indices [0," + str(self.number_of_spectra - 1) + "] ")

        if not 'Continuous' in self.GetSpectrumType():
            raise Exception(f"The ImzML format has to be in Continuous format! Current format ist {self.GetSpectrumType()}!\nUse GetSpectrum(...) instead.")

        batch_size = len(indices)    
        ys_batch = np.zeros([batch_size, self.depth], dtype=np.float32)
        idx = np.array(indices, dtype=np.uint32)

        self.lib.GetSpectra(self.handle, idx.ctypes.data_as(
            POINTER(c_uint32)), batch_size, ys_batch.ctypes.data_as(
            POINTER(c_float)))

        return ys_batch

    def GetNumberOfSpectra(self):
        self.CheckHandle()
        return self.number_of_spectra

    def SpectrumIterator(self):
        self.CheckHandle()
        for i in range(self.number_of_spectra):
            xs, ys = self.GetSpectrum(i)
            yield i, xs, ys

    def SpectrumRandomBatchIterator(self, batch_size):
        self.CheckHandle()
        while(True):
            ids = np.random.random_integers(
                0, self.number_of_spectra-1, batch_size)
            data = np.array([self.GetSpectrum(int(i))[1] for i in ids])
            yield data



    # def _NeighborsStructureElement(self,neighbors, shape):
    #     N = np.prod(shape)
    #     d = np.array(N*[0])
    #     d[:neighbors] = 1
    #     random.shuffle(d)
    #     while d[N//2] == 1:
    #         random.shuffle(d)
    #     d[N//2] = 1
    #     return np.reshape(d,shape)

    # def BatchIteratorWithNNeighbors(self, batch_size, shuffle=True, neighbors=3):
    #     """
    #         1 random entry + subsequently n neighbors
    #         [random sample x]
    #         [neighbor n-1]
    #         [neighbor n-2]
    #         ...
    #         [neighbor n-n]
    #     """
    #     assert(batch_size%(neighbors+1) == 0)

    #     ids = [i+1 for i in range(self.GetNumberOfSpectra())]
    #     pos = np.zeros((len(ids),3),np.int32)
    #     for i in ids:
    #         pos[i-1] = self.GetSpectrumPosition(i-1)

    #     spatial_representation = np.zeros(self.GetShape()[:2])
    #     spatial_representation[pos[:,0], pos[:,1]] = ids

    #     missing_values = len(ids) % batch_size
    #     # ensure full batches by padding the id list
    #     ids = np.pad(ids, (missing_values//2 + missing_values%2, missing_values//2), mode='reflect')
    #     assert((len(ids) % batch_size) != 0)

    #     structure_element = self._NeighborsStructureElement(neighbors, [3,3])

    #     if shuffle:
    #         random.shuffle(ids)

    #     number_of_seeds = batch_size // (neighbors+1)
    #     # generate batches until all ids are touched once
    #     for p in range(len(ids) // batch_size - 1):
    #         neighbor_ids = []

    #         for seed in ids[p*number_of_seeds:(p+1)*number_of_seeds]:
    #             seed_id = seed
    #             while True:
    #                 mask = ndimage.binary_dilation(spatial_representation == seed_id, structure=structure_element).astype(bool)
    #                 masking_result = spatial_representation[mask]
    #                 if len(masking_result) != (neighbors + 1):
    #                     seed_id = ids[random.randint(0,len(ids)-1)]
    #                 else:
    #                     break
    #             masking_result[masking_result == 0] = masking_result[masking_result != 0][0]
    #             neighbor_ids.append(masking_result)

    #         data = np.array([self.GetSpectrum(int(i-1))[1] for i in np.reshape(neighbor_ids,(-1))])

    #         yield data

    #     yield None
