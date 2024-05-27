<p align="center">
  <img src="https://github.com/neurodynamics-ai/travnet/blob/main/TravNet.repo.jpg" width="400"/>
</p>


# Introducing TravNet
 
TravNet is a suite of functions to fully automate the seperation of neurons from noise. It uses preprocessing to take a binary data file and convert it to a waveblob (waveforms and their corresponding principal components). The waveblobs are then sorted using a pretrained convolutional neural network, trained on human sorted batches. The output is a file containing the waveforms corresponding cluster ID (neuron template), and timestamp.

## Quick Start

You can follow the steps below to quickly get up and running with TravNet models. 

1. Create a conda env

2. In the top=level directory run:
    ```bash
    pip install -e .
    ```
3. To download model weights and sample data:
    ```bash
    python3 download_samples.py
    ```
4. Once the model weights are data are available you can run the model using the command below:
    ```bash
    python3 example_sorter.py
    ```