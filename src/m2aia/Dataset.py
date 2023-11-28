from operator import index
from os import times
from time import time
from typing import *
from . import ImageIO
import hashlib
import pathlib
import numpy as np
import random


class BaseDataSet():
    def __init__(self, images : List[ImageIO.ImzMLReader], buffer_type : str) -> None:
        self.images = images
        self.elements = []

        self.buffer_type = buffer_type
        if 'disk' in self.buffer_type: # buffer images on disk; check path exists
            self.buffer_path = pathlib.Path(self.buffer_type.split(':')[1])
            if not self.buffer_path.exists():
                raise Exception(f"The path {str(self.buffer_path)} does not exist!")
    
    def __len__(self) -> int:
        pass

    def __getitem__(self, index):
        pass

    def getitems(self, indexes: List[int]):
        pass
   


class SpectrumDataset(BaseDataSet):
    """Dataset for accession individual spectra and class labels (optional) of multiple images (m2aia.ImzMLReader objects). 
    
    The aim of the SpectrumDataset is to provide convenient access to spectra of single or multiple ImzMLReaders.
    Two access strategy exist:
    1) Spectral approach: a single spectrum is returned.
    2) Spatio-spectral: a central spectrum and corresponding neighbors are returned.

    To use multiple images a spectra depth of equal size for each image is required.

    A label mask can be provided and is used to return labels for each accessed element.
    
    To use the spatio-spectral approach, a shape element is required. The Dataset will then return 
    the spectrum embedded in neighboring spectra, i.e. if the shape tuple is shape=(5,5) the shape of a
    data entry is [B,C,H,W], with batchsize as B = 1, spectrum depth as C = len(spectrum), width 
    as W=5 and height as H=5 of the patch.

    If no shape element was provided, the Dataset will return a single spectrum with shape [B=1,C].

    If multiple elements of the Dataset should be queried at one, the SpectrumDataset.getitems(list_of_indices)
    returns a batch like object containing the elements. i.e. without a shape definition returned elements will have 
    the shape [B=len(list_of_indices), C] and with shape=(5,5) the shape [B=len(list_of_indices),C,H=5,W=5].
    This is used in m2aia.BatchGenerator.

    
    
    Complete processing examples with focus on deep learning can be found on https://github.com/m2aia/pym2aia-examples 
    
    Example usage::

        import m2aia as m2

        I = m2.ImzMLReader("path/to/imzMl/file.imzML")
        I.SetNormalization(m2.m2NormalizationTIC)
        I.SetIntensityTransformation(m2.m2IntensityTransformationSquareRoot)
        I.Execute()      

        dataset = m2.SpectrumDataset([I], shuffle=True)
        for X,Y in dataset():
            print("Spectrum", X.shape, "Class Labels", Y.shape)
            do_something(X,Y)

    """

    def find_nearest_indices(self, centroids: np.array, xaxis: np.array):
        return np.array([np.argmin(np.abs(xaxis - mz)) for mz in centroids])

    def find_subrange_indices(self, xs, center_index, tolerance, is_ppm):
        # Calculate the lower and upper bounds based on the index and tolerance
        center_value = xs[center_index]
        if is_ppm:
            tol = xs[center_index] * tolerance * 10e-6

        lower_bound = center_value - tol
        upper_bound = center_value + tol

        # Initialize search pointers
        left_index = center_index
        right_index = center_index

        # Move the left pointer to find the lower bound
        while left_index > 0 and xs[left_index - 1] >= lower_bound:
            left_index -= 1

        # Move the right pointer to find the upper bound
        while right_index < len(xs) - 1 and xs[right_index + 1] <= upper_bound:
            right_index += 1

        # print(left_index,center_index, right_index + 1, "=>", abs(left_index - (right_index+1)))
        # Extract the values within the range
        return left_index, right_index + 1


    def __init__(self,images:List[ImageIO.ImzMLReader], 
                 labeled_images:List[np.array] = None,
                 sampling_masks:List[np.array] = None, 
                 spectrum_mask_indices:List[int] = None,
                 tolerance:np.float32=None, 
                 is_tolerance_in_ppm:bool=True, 
                 label_map: Dict = None,
                 shape:Tuple = None, 
                 transform_data = None, 
                 transform_labels = None, 
                 buffer_type:str='memory', 
                 reduce_function=np.mean,
                 shuffle=False,
                 quiet_init=True)-> None:
        """_summary_

        Args:
            images (List[ImageIO.ImzMLReader]): A list of ImageIO.ImzMLReader objects
            labeled_images (List[np.array], optional): A list of labeled masks. If non the ImzMLReader.GetMaskArray is used for each image. Defaults to None.
            exclude_labels (List[np.int32]): A list of labels which are excluded.
            spectrum_mask_indices (np.array, optional): A list of indices along the x axis (indices of m/z values). If None, the whole spectra with all m/z values is loaded. Defaults to None.
            shape (Tuple, optional): The shape can be used to query a neighborhood around a given spectrum/pixel. For example if shape is set to (-1,5,5), a 5x5 neighborhood is sampled around a queried pixel position. If shape is set to (-1,) the shape size is set to either the number of indices given in the spectrum_mask_indices or is set to hew whole spectrum depth. Defaults to (-1,).
            tolerance (int, optional): if spectrum_mask_indices is used, a tolerance can be set to apply a reduce function around the indices. Defaults to 20.
            reduce_function (function, optional): Reduce function if tolerance is set. Defaults to numpy.mean.
            transforms (Function, optional): A transformation can pe applied to a given spectrum using e.g. the transforms of torchvision. Defaults to None.
            buffer_type (str, optional): During querying the images it is possible to buffer queried spectra in memory to provide a fast access to upcoming queries e.g. in the next epoch. Defaults to 'memory'. Disable by setting it to None.
        """
        super().__init__(images, buffer_type)

        # track member variables        
        self.shape = shape
        self.spectrum_mask_indices = spectrum_mask_indices
        self.tolerance = tolerance
        self.is_ppm = is_tolerance_in_ppm
        self.x_hws = 0
        self.y_hws = 0
        self.xs = self.images[0].GetXAxis()
        self.reduce_function=reduce_function
        self.ranges = None
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.label_map = label_map
        self.labels = set()
        
        
        

        if self.tolerance and self.spectrum_mask_indices is not None:
            self.ranges = [self.find_subrange_indices(self.xs, index, self.tolerance, self.is_ppm) for index in self.spectrum_mask_indices]
        else:
            self.tolerance = None
            self.ranges = None


        # make sure all images have identically x axis
        for imageID, handle in enumerate(self.images):          
            assert(np.all(self.xs == handle.GetXAxis())) # check for equal x axis size
        
        if self.spectrum_mask_indices is None: # complete spectrum data
            self.spectrum_depth = self.images[0].GetXAxisDepth()
        else: # mask a spectrum using a list of indices
            self.spectrum_depth = self.spectrum_mask_indices.shape[0]

            
        if self.shape and len(self.shape) >= 2:
            if self.shape[-2]%2 == 0 or self.shape[-1]%2 == 0:
                raise Exception(f"We only support odd neighborhood sizes!")          
            
            # half window size 
            self.x_hws = self.shape[-1]//2
            self.y_hws = self.shape[-2]//2

        if "memory" == self.buffer_type:
            self.buffer = []            
            for k in range(len(self.images)):
                buffer_spectrum_label = np.zeros((self.images[k].GetNumberOfSpectra(),), dtype=bool)
                buffer_spectrum_data = np.zeros((self.images[k].GetNumberOfSpectra(), self.spectrum_depth), dtype = np.float32)
                self.buffer.append((buffer_spectrum_label, buffer_spectrum_data))
        
        # self.neighborhood_size = neighborhood_size
        self.index_images = []
        self.hit_counter = [0]*len(self.images)


        # for each image
        for imageID, handle in enumerate(self.images):
            imageElements = []

            index_image = -np.ones(handle.GetShape()[::-1], dtype=np.int32)
        
            if labeled_images is None:
                mask_image = handle.GetMaskArray()
            else:
                mask_image = labeled_images[imageID]

            if sampling_masks is None:
                sampling_mask = handle.GetMaskArray()
            else:
                sampling_mask = sampling_masks[imageID]

            for spectrumID in range(handle.GetNumberOfSpectra()):
                (x,y,z) = handle.GetSpectrumPosition(spectrumID)
                
                if sampling_mask[z,y,x] <= 0 or mask_image[z,y,x] < 0:
                    continue
                index_image[z,y,x] = spectrumID
                label = mask_image[z,y,x]
                imageElements.append((imageID, spectrumID, (x,y,z), label))
                self.labels.add(label)
                

            # for each imageID insert elements and an index image
            self.elements.extend(imageElements)
            self.index_images.append(index_image)
    

        if shuffle:
            random.shuffle(self.elements)

    def __len__(self) -> int:
        """
        Returns the number of accessible spectra.
        If multiple images are used, len is the sum of accessible elements of all images.
        
        Returns:
            int: Number of accessible elements of this Dataset.
        """
    
        return len(self.elements)

    def __getitem__(self, index):
        
        return self.getitems([index])
    

    def getitems(self, dataset_query_indices: List[int]):
        
        image_query_indices = {imageID:[] for imageID, _ in enumerate(self.images)}
        image_query_labels = {imageID:[] for imageID, _ in enumerate(self.images)}

        for index in dataset_query_indices:
            # check origin
            imageID, spectrumID, (x,y,z), label = self.elements[index]
            image_shape = self.images[imageID].GetShape()
            # save label
            image_query_labels[imageID].append(label)
            if self.shape:
                left = np.clip(x-self.x_hws, 0, image_shape[0])
                right = np.clip(x+self.x_hws+1, 0, image_shape[0])
                bottom = np.clip(y-self.y_hws,0, image_shape[1])
                top = np.clip(y+self.y_hws+1,0, image_shape[1])
                indices = self.index_images[imageID][0,bottom:top,left:right].flatten()
                if any(indices < 0): 
                    choices = np.random.choice(indices[indices>=0], len(indices[indices < 0]))
                    indices[indices < 0] = choices
                
                expected_indices = (2*self.x_hws+1) * (2*self.y_hws+1)
                if len(indices) < expected_indices:
                    missing_indices = expected_indices - len(indices)
                    indices = np.concatenate([indices,np.random.choice(indices, missing_indices)])

                image_query_indices[imageID].append(indices)
            else:
                image_query_indices[imageID].extend([spectrumID])

        BUFFER_QUERY = 0
        BUFFER_DATA = 1
        result_data = None
        result_labels = None

        for imageID, image in enumerate(self.images):
            # get all indices for the image
            indices = np.array(image_query_indices[imageID], dtype=np.int32)
            labels = np.array(image_query_labels[imageID], dtype=np.int32)
            

            if self.buffer_type == "memory": # buffering is used
                image_buffer_data = self.buffer[imageID][BUFFER_DATA]
                image_buffer_query = self.buffer[imageID][BUFFER_QUERY]
                
                # mask indices which require to be loaded from ImzML image (buffer status False indicates "not buffered")
                query_mask = image_buffer_query[indices] == False
                miss_indices = indices[query_mask]
                # hit_indices = image_indices[~query_mask]

                # load from imzML and store in buffer
                if np.any(query_mask):
                    # print(len(miss_indices)/len(indices))

                    # get data for those indices from the image
                    spectra = image.GetSpectra(miss_indices).astype(np.float32)
                    image_buffer_query[miss_indices] = True

                    if self.spectrum_mask_indices is None: # check if a centroids list exists
                        # no one was set so we put the raw spectra into the buffer
                        image_buffer_data[miss_indices] = spectra
                    else: # a spectrum mask exists
                        if self.ranges: # use range queries along the spectra
                            for k, [l,u] in enumerate(self.ranges):
                                image_buffer_data[miss_indices,k] = self.reduce_function(spectra[:,l:u],axis=1)
                                # print(self.reduce_function(spectra[:,l:u],axis=1), l,u, image_buffer_data[miss_indices][:,k])
                        else:
                            image_buffer_data[miss_indices] = spectra[..., self.spectrum_mask_indices]
            
                # load from buffer           
                batch_data = image_buffer_data[indices]
                # print("mean", np.mean(batch_data))
            else: # no buffering is used
                spectra = self.images[imageID].GetSpectra(indices)
                if self.spectrum_mask_indices is not None:
                    if self.ranges: # use range queries along the spectra
                        for k, [l,u] in enumerate(self.ranges):
                            batch_data = self.reduce_function(spectra[:,l:u],axis=1)
                    else:
                        batch_data = spectra[..., self.spectrum_mask_indices]
                else:
                    batch_data = self.images[imageID].GetSpectra(indices)            

            if self.shape: # reshape to [B,C,H,W]
                batch_data = np.reshape(batch_data, (-1,) + self.shape + (self.spectrum_depth,))
                batch_data = np.swapaxes(batch_data, 1,3)
                
            
            # combine queries from different images
            if result_data is None:
                result_data = batch_data
            else:
                result_data = np.concatenate([result_data, batch_data])

            if result_labels is None:
                result_labels = np.array(labels, np.int32)
            else:
                result_labels = np.concatenate([result_labels, np.array(labels, np.int32)])
            
            
        
        # at this stage we receive a list of spectra for n queried central spectra we receive a list of
        # n*(2*s+1)^2 where s is the half size of the neighborhood
        # we can add a transformer that now reshapes those spectra into the correct shape.
        # e.g. from [n*(2*s+1)^2,D] we go to => [n, D, 2*s+1, 2*s+1]
        # with D is the number of channels

        # trafo = transforms.Compose([
        #     transforms.Lambda(lambda x: np.transpose(x)), #=> (D,9)
        #     transforms.Lambda(lambda x: np.reshape(x, (1, x.shape[0], int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1]))))), #=> (1,D,3,3)
        #     transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))])
        if self.transform_data is not None:
            result_data = self.transform_data(result_data)
        
        if self.transform_labels is not None:
            result_labels = self.transform_labels(result_labels)
        
        return result_data, result_labels



class IonImageDataset(BaseDataSet):
    def __init__(self, images : List[ImageIO.ImzMLReader], 
                       centroids:List[float], 
                       tolerance:float, 
                       tolerance_type:str='ppm', 
                       buffer_type='memory', 
                       transforms=None)-> None:
        super().__init__(images, buffer_type)
        self.elements = centroids
        self.tolerance = tolerance
        self.tolerance_type = tolerance_type
        self.transforms = transforms
        self.buffer = [{} for _ in images]
        


    def get_tolerance(self, c): 
        if self.tolerance_type == "ppm":
            return c * self.tolerance * 10e-6
        else:
            return self.tolerance

    def make_buffered_image(self, index):
        c = self.elements[index%len(self.elements)]
        image_id = index//len(self.elements)
        if c not in self.buffer[image_id] or self.buffer_type == 'none':
            # create ion image
            ii = self.images[image_id].GetArray(c, self.get_tolerance(c), squeeze=False)
            
            if self.buffer_type == 'memory': # buffer image in memory
                self.buffer[image_id][c] = ii
            else: # buffer image on disk
                hash = hashlib.sha224("{:.8f}".format(c).encode())
                hash.update("{:.8f}".format(self.get_tolerance(c)).encode())
                path = self.buffer_path.joinpath(self.images[image_id].GetImageName())
                path.mkdir(exist_ok=True)
                path = path.joinpath(hash.hexdigest())
                np.save(path ,ii)
                self.buffer[image_id][c] = path

        else: # load buffered entry
            if self.buffer_type == 'memory':
                ii = self.buffer[image_id][c]
            else:
                ii = np.load(str(self.buffer[image_id][c])+".npy")
        
        if self.transforms is not None:
            ii = self.transforms(ii)
        return ii

    def __len__(self) -> int:
        return len(self.elements) * len(self.images)

    def __getitem__(self, index):
        return self.make_buffered_image(index)

    def getitems(self, indexes: List[int]):
        return np.stack([self.__getitem__(index) for index in indexes])

