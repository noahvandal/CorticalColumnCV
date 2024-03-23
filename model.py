'''
3/23/2024
Noah Vandal

Thought: training a model using cortical columns for parsing through the image may be useful for better modelling how the brain works. 
'''
import torch
import torch.nn as nn
from .cortical_column import MixtureOfCorticalColumns
import torch.nn.functional as F

class CorticalModel(nn.Module):
    def __init__(self, image_shape, num_columns, receptive_paradigms=8, min_kernel_dim=16, max_kernel_dim=256, num_layers=2, target_shape=16):
        super(CorticalModel, self).__init__()
        self.num_columns = num_columns
        self.image_shape = image_shape
        self.rece = receptive_paradigms


        # init cortical columns
        self.column_mix = MixtureOfCorticalColumns(image_shape, num_columns, receptive_paradigms, min_kernel_dim, max_kernel_dim, num_layers, target_shape)

    def cropReceptiveField(self, receptive_field, image):
        return image[receptive_field[0]:receptive_field[1], receptive_field[2]:receptive_field[3]]
    

    def reconstructImage(self, outputList):
        # Assuming image_shape is (H, W) for grayscale or (C, H, W) for RGB
        channels = 1 if len(self.image_shape) == 2 else self.image_shape[0]
        height, width = self.image_shape[-2], self.image_shape[-1]
        image = torch.zeros((channels, height, width))

        for output, receptive_field in outputList:
            start_row, end_row, start_col, end_col = receptive_field
            target_height = end_row - start_row
            target_width = end_col - start_col

            # Resizing output to match the receptive field size in the original image
            resized_output = F.interpolate(output.unsqueeze(0),  # Add batch dimension
                                           size=(target_height, target_width), 
                                           mode='bilinear',  # or 'nearest' for categorical
                                           align_corners=False).squeeze(0)  # Remove batch dimension

            # Placing the resized output in the correct location
            if channels > 1:  # For multichannel images
                image[:, start_row:end_row, start_col:end_col] = resized_output
            else:  # For grayscale images
                image[start_row:end_row, start_col:end_col] = resized_output

        return image
    
    def paradigmPass(self, index, image):
        receptive_fields = self.column_mix.getReceptiveFields(index)
        columns = self.column_mix.getColumns(index)

        outputs = []

        for i in range(len(columns)):
            output = columns[i].forward(self.cropReceptiveField(receptive_fields[i], image))
            outputs.append([output, receptive_fields[i]])
        
        return outputs
    
    def forward(self, image):
        imageList = []

        for i in range(self.rece):
            outputList = self.paradigmPass(i, image)  ## could do loss on each cortical column here, in embeddign space
            imageList.append(self.reconstructImage(outputList))  # could also do loss on the reconstructred image here
        

        # could combine the images and output lists, or just return the image list


        # when training can call the weight average update to average weigths with respet to nearby columns