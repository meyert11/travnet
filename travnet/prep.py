import numpy as np
import os
from dataclasses import dataclass
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from tkinter import filedialog
from sklearn.decomposition import PCA

@dataclass
class DataArgs:
    filename: str = None
    path: str = None
    num_channels: int = 24
    sample_rate: int = 30000
    threshold: int = -8

class Prep:
    def __init__(self, data_args: DataArgs):        
        if data_args.filename is not None:
            self.filename = data_args.filename
            self.num_channels = data_args.num_channels
            print(f'Filename: {DataArgs.filename}')
        else:
            print('Please select a file to import. Tip: use the window, which sometimes appears beind the current windows on the desktop.')
            current_directory = os.getcwd()
            full_path = filedialog.askopenfilename(initialdir=current_directory)
            DataArgs.path, DataArgs.filename = os.path.split(full_path)

            print("Directory Path:", DataArgs.path)  # prints the directory path
            print("Filename:", DataArgs.filename)  # prints the filename            
            self.filename = full_path
            self.num_channels = data_args.num_channels
        return None

    def import_data(self):
        # Load a binary datafile with known number of channels and place in an array
        # where rows is the channels and columns are the voltage samples
        # Open the file and read the data
        with open(self.filename, 'rb') as f:
            data = np.fromfile(f, dtype=np.int16)

        # Reshape the data so that channels are along the rows
        num_samples = data.size // self.num_channels
        data = data.reshape((self.num_channels, num_samples))

        return data
    
    def filter_data(self, signal):
        # Bandpass filtering of broadband signal to remove lower 250 Hz noise
        # Define the filter parameters        
        cutoff_freq = 250  # Hz
        order = 4

        # Calculate the normalized cutoff frequency
        nyquist_freq = 0.5 * self.sample_rate
        low = cutoff_freq / nyquist_freq

        # Design the Butterworth filter
        b, a = butter(order, low, btype='high')

        # Apply the filter to the data
        dataF = np.empty_like(signal)
        for i in tqdm(range(signal.shape[0]), desc="Filtering data"):
            dataF[i] = filtfilt(b, a, signal[i])

        return dataF

        return dataF
    
    def getWFs(self, signal):
        # Use threshold crossings and valley to identify waveforms from broadband signal
        # Get the waveform indices        
        threshold = float(self.threshold)
        
        # Find the indices where the waveform drops below the threshold
        #below_threshold_indices = [i for i, value in enumerate(filtered_data) if value < threshold]

        # find threshold where the below_threshold_indices are less than 4500000
        
        idx = np.where(signal < threshold)
        below_threshold_indices = idx[0].tolist()
        while len(below_threshold_indices) > 3500000:
            threshold -= 0.3
            print(f'Finding Threshold: {threshold}')
            idx = np.where(signal < threshold)
            below_threshold_indices = idx[0].tolist()    

        waveforms = []
        total_iterations = len(below_threshold_indices)
        timestamps = []
        #TODO progress bar doesn't account for the fact that the index is not incremented for every iteration
        with tqdm(total=total_iterations, desc="Sorting Waveforms") as pbar:
            # Remove 32 datapoints from the filtered_data if the waveform drops below the threshold
            index = 0
            lastwf = -24 # index for where to start on the first potential waveform
            while index < len(below_threshold_indices):                
                if index > lastwf:
                    if index < len(signal) - 32:

                        s_index = np.argmin(signal[index:index+32])    
                        index = index + s_index                        
                        waveforms = np.append(waveforms, signal[index-8:index+24])
                        timestamps = np.append(timestamps, index)
                        lastwf = index + 24
                        
                
                index += 1
                pbar.update(1)
        waveforms = np.reshape(waveforms, (int(len(waveforms)/32), 32))
        
        print(f'Total waveforms: {len(waveforms)}')
        
        return waveforms, timestamps
    
    def PCA(waveforms):               
        # Compute the 3 principal components of the waveforms data
        # Create an instance of PCA with n_components=2
        pca = PCA(n_components=3)

        # Fit the PCA model to the waveforms data
        pca.fit(waveforms)

        # Transform the waveforms data to the first two principal components
        waveforms_pca = pca.transform(waveforms)

        # Access the first two principal components
        principal_component_1 = waveforms_pca[:, 0]
        principal_component_2 = waveforms_pca[:, 1]
        principal_component_3 = waveforms_pca[:, 2]
        
        return principal_component_1, principal_component_2, principal_component_3

    def scaleUV(waveforms):
            # input file with array of nx32 where the values are in microvolts and output an array of nx32x32 where the micrvolts are scaled to 0-31 for the 2nd dimension
            data_scaled = np.zeros((waveforms.shape[0], 32, 32), dtype=np.int16)

            #TODO check the x,y dimensions
            # Scale each row to be between 0 and 31
            for i in range(waveforms.shape[0]):
                row = waveforms[i, :]
                scaled_row = 31 * (row - row.min()) / (row.max() - row.min())
                data_scaled[i, :, :] = np.round(scaled_row).astype(np.int16)

            return data_scaled

    def ImGen(data_in, pc1, pc2, pc3):
        # Get the number of samples
        n = data_in.shape[0]

        # Initialize a new 4D array with dimensions nx3x32x32
        data_out = np.zeros((n, 3, 32, 32))

        # For each sample, set the 3D array at the corresponding location
        for i in range(n):
            data_out[i, 0, :, :] = data_in[i, :, :] * pc1[i]
            data_out[i, 1, :, :] = data_in[i, :, :] * pc2[i]
            data_out[i, 2, :, :] = data_in[i, :, :] * pc3[i]

        return data_out.astype(float)