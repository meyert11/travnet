<p align="center">
  <img src="https://github.com/neurodynamics-ai/travnet/blob/main/TravNet.repo.jpg" width="400"/>
</p>


# Introducing TravNet
 
TravNet is a suite of functions to fully automate the seperation of Neurons from Noise. It uses preprocessing to take a binary data file and convert it to a waveblob (waveforms and their corresponding principal components). The waveblobs are then sorted using a pretrained convolutional neural network, trained on human sorted batches. The output is a file containing the waveforms corresponding cluster ID (neuron template), and timestamp.

