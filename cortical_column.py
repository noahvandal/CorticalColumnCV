"""
3/23/2024
Noah Vandal

Implementation of brain inspired cortical columns to generate random receptive fields that can then be trained on. 
The thought is that having a leaner, sparse architecure, but the ability to hone in on specific features and receptive fields
may allow for faster processing and better generalization.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from itertools import combinations


class CorticalColumn(nn.Module):
    def __init__(self, kernel_dim, num_layers=2, target_shape=None):
        super(CorticalColumn, self).__init__()
        self.kernel_dim = kernel_dim
        self.kernel = nn.Parameter(torch.randn(kernel_dim, kernel_dim))
        self.bias = nn.Parameter(torch.randn(1))
        self.activation = nn.ReLU()
        self.num_layers = num_layers
        self.target_shape = target_shape

    def singleLayer(self, input):
        return self.activation(torch.matmul(input, self.kernel) + self.bias)

    def forward(self, input):
        # Resize input to target shape if specified (thought: each cortical column has the same shape, allowing high level and low level features to get the same amount of attention)
        if self.target_shape is not None:
            # Assuming input is a 4D tensor (B, C, H, W) and target_shape specifies the desired H and W
            input = F.interpolate(
                input, size=self.target_shape, mode="bilinear", align_corners=False
            )

        for _ in range(self.num_layers):
            input = self.singleLayer(input)

        return input


def pseudoRandomReceptiveFieldGenerator(
    image_shape, number_of_columns, min_size, max_size
):
    """
    Generate random receptive fields, ensuring that the entire image is covered at least once.
    """
    height, width = image_shape

    receptive_fields = []
    coverage_map = np.zeros(image_shape, dtype=bool)

    # Generate initial random receptive fields
    for _ in range(number_of_columns):
        kernel_height = random.randint(min_size, max_size)
        kernel_width = random.randint(min_size, max_size)

        # Ensure kernel fits within the image
        start_row = random.randint(0, height - kernel_height)
        start_col = random.randint(0, width - kernel_width)

        receptive_fields.append(
            (
                (start_row, start_col),
                (start_row + kernel_height, start_col + kernel_width),
            )
        )
        coverage_map[
            start_row : start_row + kernel_height, start_col : start_col + kernel_width
        ] = True

    # Check for and cover any uncovered areas
    uncovered_indices = np.where(coverage_map == False)
    uncovered_points = list(zip(uncovered_indices[0], uncovered_indices[1]))

    while uncovered_points:
        point = random.choice(uncovered_points)
        kernel_height = random.randint(min_size, min(max_size, height - point[0]))
        kernel_width = random.randint(min_size, min(max_size, width - point[1]))

        start_row, start_col = point
        receptive_fields.append(
            (
                (start_row, start_col),
                (start_row + kernel_height, start_col + kernel_width),
            )
        )

        # Update coverage map and uncovered points list
        coverage_map[
            start_row : start_row + kernel_height, start_col : start_col + kernel_width
        ] = True
        uncovered_indices = np.where(coverage_map == False)
        uncovered_points = list(zip(uncovered_indices[0], uncovered_indices[1]))

    return receptive_fields


class MixtureOfCorticalColumns(nn.Module):
    def __init__(
        self,
        image_shape,
        target_number_of_columns,
        number_of_receptive_paradigms,
        min_size,
        max_size,
        num_layers=2,
        target_receptive_size=16,
    ):
        """
        Generate random receptive fields; multiple paradigms of receptive fields are generated, each with a different set of receptive fields and columns,
        and the input is passed through each paradigm. The paradigm that performs better can be selected for the final output.
        """
        super(MixtureOfCorticalColumns, self).__init__()
        self.image_shape = image_shape
        self.number_of_columns = target_number_of_columns
        self.number_of_receptive_paradigms = number_of_receptive_paradigms
        self.min_size = min_size
        self.max_size = max_size
        self.num_layers = num_layers
        self.target_receptive_size = target_receptive_size

        for i in range(number_of_receptive_paradigms):
            receptive_fields, columns = self.generateReceptiveFieldParadigm()
            self.__setattr__(f"receptive_fields_{i}", receptive_fields)
            self.__setattr__(f"columns_{i}", columns)

        self.fieldList = {
            f"receptive_fields_{i}": self.__getattr__(f"receptive_fields_{i}")
            for i in range(number_of_receptive_paradigms)
        }
        self.columnList = {
            f"columns_{i}": self.__getattr__(f"columns_{i}")
            for i in range(number_of_receptive_paradigms)
        }

    def getReceptiveFields(self, paradigm_index):
        return self.__getattr__(f"receptive_fields_{paradigm_index}")

    def getColumns(self, paradigm_index):
        return self.__getattr__(f"columns_{paradigm_index}")

    def generateReceptiveFieldParadigm(self):
        receptive_fields = pseudoRandomReceptiveFieldGenerator(
            self.image_shape, self.number_of_columns, self.min_size, self.max_size
        )
        number_of_columns = len(
            receptive_fields
        )  # update in case more receptive fields were generated

        columns = nn.ModuleList(
            [
                CorticalColumn(
                    kernel_dim=self.max_size - self.min_size + 1,
                    num_layers=self.num_layers,
                    target_shape=(
                        self.target_receptive_size,
                        self.target_receptive_size,
                    ),
                )
                for _ in range(number_of_columns)
            ]
        )

        return receptive_fields, columns

    def calculateProximity(self, rf1, rf2):
        """
        Calculate the Euclidean distance between the upper left points of two receptive fields.
        """
        # Using only the upper left points (start_row, start_col) for both receptive fields
        return np.sqrt((rf1[0][0] - rf2[0][0]) ** 2 + (rf1[0][1] - rf2[0][1]) ** 2)

    def averageWeightsBasedOnProximity(self, threshold_distance=50):
        """
        Averages the weights of cortical columns based on the proximity of their receptive fields.
        Proximity is defined by the Euclidean distance between the upper left points of the receptive fields.
        Columns whose receptive fields are within the 'threshold_distance' of each other are considered nearby.
        """
        # Iterate through each paradigm
        for paradigm_index in range(self.number_of_receptive_paradigms):
            columns = self.getColumns(paradigm_index)
            receptive_fields = self.getReceptiveFields(paradigm_index)

            # List to store average kernels and biases for columns considered in averaging
            average_kernels = [torch.zeros_like(column.kernel) for column in columns]
            average_biases = [torch.zeros_like(column.bias) for column in columns]
            counts = [0] * len(columns)  # Count of nearby columns for averaging

            # Calculate distances and average for nearby columns
            for i, j in combinations(range(len(columns)), 2):
                distance = self.calculateProximity(
                    receptive_fields[i], receptive_fields[j]
                )
                if distance <= threshold_distance:
                    average_kernels[i] += columns[j].kernel
                    average_biases[i] += columns[j].bias
                    counts[i] += 1

                    # Symmetrically averaging
                    average_kernels[j] += columns[i].kernel
                    average_biases[j] += columns[i].bias
                    counts[j] += 1

            # Update kernels and biases where averages were computed
            for idx, column in enumerate(columns):
                if counts[idx] > 0:  # Ensure division by zero is avoided
                    column.kernel.data = (column.kernel + average_kernels[idx]) / (
                        counts[idx] + 1
                    )
                    column.bias.data = (column.bias + average_biases[idx]) / (
                        counts[idx] + 1
                    )
