import torch 
# from .. import BatchGenerator, ImzMLReader

# class SpectrumBatchDataset(torch.utils.data.Dataset):
#     def __init__(self, image_handle: ImzMLReader, batch_size: int, shuffle: bool=True):
#         super(SpectrumBatchDataset,self).__init__()
#         self.gen = BatchGenerator(image_handle, batch_size, shuffle)
    
#     def __len__(self):
#         return self.gen.__len__()

#     def on_epoch_end(self):
#         print("Epch ends")
#         self.gen.on_epoch_end()
    
#     def __getitem__(self, index):
#         if (index+1)*self.batch_size == len(self):
#             self.on_epoch_end()
#         return self.gen.__getitem__(index)


# class MultiImageSpectrumBatchDataset(torch.utils.data.Dataset):
#     def __init__(self, image_handles, batch_size: int, shuffle: bool=True):
#         super(MultiImageSpectrumBatchDataset,self).__init__()
#         self.gen = MultiImageSpectrumBatchGenerator(image_handles, batch_size, shuffle)
    
#     def __len__(self):
#         return self.gen.__len__()

#     def on_epoch_end(self):
#         print("Epch ends")
#         self.gen.on_epoch_end()
    
#     def __getitem__(self, index):
#         if (index+1)*self.batch_size == len(self):
#             self.on_epoch_end()
#         return self.gen.__getitem__(index)