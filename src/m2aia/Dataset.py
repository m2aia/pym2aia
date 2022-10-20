from operator import index
from os import times
from time import time
from typing import *
from . import ImageIO
import hashlib
import pathlib
import numpy as np


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
    """Dataset for accession of individual spectra. 
    
    __len__ so that len(dataset) returns the size of the dataset, that is equal to the sum of the number of spectra (N) for each image.
    __getitem__ to support the indexing such that dataset[i] can be used to get i'th sample. i is pointing to indices 0,...,p-1,p,...,q-1, ... N, with p=#SpectraImage1, q=#SpectraImage2 etc...
    neighborhood_size so that 2*neighborhood_size+1 is the window size

    Parameters
    ----------
    images : List[ImageIO.ImzMLReader]
       
    """

    def __init__(self,images : List[ImageIO.ImzMLReader], neighborhood_size: int = 0, transforms = None, buffer_type='memory')-> None:
        super().__init__(images, buffer_type)
        
        self.spectrum_depth = self.images[0].GetXAxisDepth()
        if "memory" == self.buffer_type:
            self.buffer = [(np.array([None] * self.images[k].GetNumberOfSpectra()), np.zeros((self.images[k].GetNumberOfSpectra(), self.spectrum_depth), dtype = np.float32)) for k in range(len(self.images))]
        
        self.neighborhood_size = neighborhood_size
        self.index_images = []
        self.footprint = np.ones([2*neighborhood_size+1, 2*neighborhood_size+1, 1])
        self.transforms = transforms

        self.hit_counter = [0]*len(self.images)

        for imageID, handle in enumerate(self.images):
            assert(self.spectrum_depth == handle.GetXAxisDepth())
            imageElements = []
            index_image = handle.GetArray(handle.GetXAxis()[self.spectrum_depth//2], 10).astype(np.int32)
            index_image.fill(-1)

            for spectrumID in range(handle.GetNumberOfSpectra()):
                (x,y,z) = handle.GetSpectrumPosition(spectrumID)
                imageElements.append((imageID, spectrumID, (x,y,z)))
                index_image[z,y,x] = spectrumID

            self.elements.extend(imageElements)
            self.index_images.append(index_image)

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, index):
        return self.getitems([index])

    def getitems(self, indexes: List[int]):
        ids_split_to_images = {}

        #sort by images and create list of image related ids         
        for index in indexes:
            imageID, spectrumID, (x,y,z) = self.elements[index]
            shape = self.images[imageID].GetShape()
            if imageID not in ids_split_to_images:
                ids_split_to_images[imageID] = []
            
            if self.neighborhood_size > 0: 
                # neighborhood do not violate border regions
                left = np.clip(x-self.neighborhood_size, 0, shape[0])
                right = np.clip(x+self.neighborhood_size+1, 0, shape[0])
                bottom = np.clip(y-self.neighborhood_size,0, shape[1])
                top = np.clip(y+self.neighborhood_size+1,0, shape[1])

                # get all spectra indices for the requested spectrum position and neighbors
                indices = self.index_images[imageID][0,bottom:top,left:right]
                indices = indices.flatten()

                # handle invalid spectra positions (indicated by -1 in index images)
                if np.any(indices < 0): 
                    choices = np.random.choice(indices[indices>=0], len(indices[indices < 0]))
                    indices[indices < 0] = choices

                # handling missing values if at border
                expected_indices = (self.neighborhood_size*2+1)**2
                if len(indices) < expected_indices:
                    missing_indices = expected_indices - len(indices)
                    indices = np.concatenate([indices,np.random.choice(indices, missing_indices)])

                indices = indices.tolist()
            else:
                indices = [spectrumID]

            ids_split_to_images[imageID].extend(indices)

        
        
        result = None
        interim = None
        BUFFER_QUERY = 0
        BUFFER_DATA = 1
        for imageID, indices in ids_split_to_images.items():
            
            # use buffering
            if self.buffer_type is not None:

                indices = np.array(indices)
                interim = np.zeros((len(indices), self.spectrum_depth))

                # mask buffer entries which are not already filled with data
                query_mask = self.buffer[imageID][0][indices] == None
                
                # load data from imzML
                if query_mask.any():
                    self.hit_counter[imageID] = self.hit_counter[imageID] + 1
                    print(self.hit_counter, end='\r')
                    # get data for those indices from the image
                    interim[query_mask] = self.images[imageID].GetSpectra(indices[query_mask])
                    # mark buffer query structure to prevent double queries
                    self.buffer[imageID][BUFFER_QUERY][indices[query_mask]] = True

                    # copy data to buffer
                    self.buffer[imageID][BUFFER_DATA][indices[query_mask]] = interim[query_mask]
                    
                # for all entries in the BUFFER_QUERY structure which already have data
                # copy buffered data to interim data structure
                if ~query_mask.any():
                    interim[~query_mask] = self.buffer[imageID][BUFFER_DATA][indices[~query_mask]]
            else:
                interim = self.images[imageID].GetSpectra(indices)
            
            if result is None:
                result = interim
            else:
                result = np.concatenate([result, interim])
        
        if self.transforms is not None:
            result = self.transforms(result)
        
        return result



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
            ii = self.images[image_id].GetArray(c, self.get_tolerance(c))
            
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

