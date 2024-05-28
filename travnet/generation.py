
import time
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from .prep import Prep
from .model import CNN


class TravNet:
    def __init__(self, data_args):
        self.sample_rate = data_args.sample_rate        
        self.threshold = data_args.threshold
        
    def prepare_data(self, spk_dataset):
        # Filter the data                                
        start_time = time.time()
        filtered = Prep.filter_data(self, spk_dataset)
        print(f"Filtered in {time.time() - start_time:.2f} seconds")

        # Get the waveforms and timestamps        
        start_time = time.time()
        #TODO - add a progress bar here and run through each channel
        waveforms, timestamps = Prep.getWFs(self, filtered[0,:])
        print(f"Got waveforms in {time.time() - start_time:.2f} seconds")
        
        # Perform PCA on the waveforms
        pc1, pc2, pc3 = Prep.PCA(waveforms)
        # Scale the waveforms
        scaled_waveforms = Prep.scaleUV(waveforms)
        # Generate the image
        waveblob = Prep.ImGen(scaled_waveforms, pc1, pc2, pc3)
        waveblob = waveblob.astype(float)
        
        # Create the DataLoader
        eval_loader = DataLoader(waveblob, batch_size=40000)
        print('Data prepared')
        return eval_loader, waveblob, timestamps, pc1, pc2

    @torch.inference_mode()
    def spike_sorter(self, eval_loader):
        # Load the model
        model_weights = self.filename
        print(f"Loading model from {model_weights}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN().to(device)
        model.load_state_dict(torch.load(model_weights, map_location=torch.device(device))['state_dict'])
        
        outputs = []
        # Run Model
        for inputs in eval_loader:
            inputs = inputs.float().to(device)
            with torch.no_grad():
                output = model(inputs)
            outputs.append(output)

        outputs = torch.cat(outputs)
        return outputs
    
    def process_outputs(outputs, timestamps, pc1, pc2):
        # Find the argmax of outputs and call it clusters
        clusters = np.argmax(outputs, axis=1).cpu().numpy()

        # Create a DataFrame with clusters, timestamps, pc1, and pc2 as columns
        df = pd.DataFrame({
            'clusters': clusters,
            'timestamps': timestamps,
            'pc1': pc1,
            'pc2': pc2
        })

        return df