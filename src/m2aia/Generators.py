import random
from . import Dataset

class BatchGenerator():

    def __init__(self, dataset: Dataset.BaseDataSet, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.elements = [i for i in range(len(self.dataset)//self.batch_size)]
        # missing_values = self.batch_size-(len(self.elements) % self.batch_size)
        # self.elements.extend(random.sample(self.elements, missing_values))
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.dataset.elements)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.elements)

    def __getitem__(self, index):
        batch_indices = list(range(index*self.batch_size,(index+1)*self.batch_size,1))
        # print("batch_indices", batch_indices)
        data, labels = self.dataset.getitems(batch_indices)
        return data, labels


# class IonImageBatchGenerator():

#     def __init__(self, dataset: Dataset.IonImageDataset, batch_size: int, shuffle: bool = True):
        
#         self.dataset = dataset
        

#         if tolerance_type == "ppm":
#             self.tolerance = self.tolerance * 10e-6

#         if elements is not None:
#             self.elements = elements
#         else:
#             self.elements = image_handle.GetXAxis().tolist()
#             print("WARNING!", "All m/z bins are used to generate ion images", f"[{len(self.elements)}]")

#         self.buffer_type = buffer_type
#         self.buffer_path = None
#         self.buffer = {}
#         if 'disk' in self.buffer_type: # buffer images on disk; check path exists
#             self.buffer_path = pathlib.Path(self.buffer_type.split(':')[1])
#             if not self.buffer_path.exists():
#                 raise Exception(f"The path {str(self.buffer_path)} does not exist!")

        
#         missing_values = self.batch_size-(len(self.elements) % self.batch_size)
#         self.elements.extend(random.sample(self.elements, missing_values))
        
#         self.on_epoch_end()

#     def on_epoch_end(self):
#         if self.shuffle:
#             random.shuffle(self.elements)

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return len(self.elements) // self.batch_size

#     def get_tolerance(self, c): 
#         if self.tolerance_type == "ppm":
#             return c * self.tolerance * 10e-6
#         else:
#             return self.tolerance

#     def make_buffered_image(self, c):
#         if c not in self.buffer or self.buffer_type == 'none':
#             # create ion image
#             ii = self.image_handle.GetArray(c, self.get_tolerance(c), self.dtype)
            
#             if self.buffer_type == 'memory': # buffer image in memory
#                 self.buffer[c] = ii
#             else: # buffer image on disk
#                 hash = hashlib.sha224("{:.8f}".format(c).encode())
#                 hash.update("{:.8f}".format(self.get_tolerance(c)))
#                 path = self.buffer_path.joinpath(self.image_handle.GetImageName())
#                 path.mkdir(exist_ok=True)
#                 path = path.joinpath(hash.hexdigest())
#                 np.save(path ,ii)
#                 self.buffer[c] = path
            
#             return ii

#         else: # load buffered entry
#             if self.buffer_type == 'memory':
#                 return self.buffer[c]
#             else:
#                 return np.load(str(self.buffer[c])+".npy")
        

#     def __getitem__(self, index):
#         X = np.array([self.make_buffered_image(c) for c in self.elements[index*self.batch_size:(index+1)*self.batch_size]])
        
#         return X
