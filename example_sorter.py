# Copyright (c) Neurodynamics Sollutions llc.
# See LICENSE file for licensing details.

import numpy as np

import fire

from travnet import TravNet, CNN, ModelArgs, Prep, DataArgs

def main(
        #TODO add arguments here
    #ckpt_dir: str,
    #data_fn: str,
    #model_path: str = ##TODO convert model to file json
    #num_channels: int = 0,
    #sample_rate: int = 30000,
    #threshold: float = -8.0,
    batch_size: int = 40000    
):
    #Load the data
    data = Prep(DataArgs).import_data()

    data_out = TravNet(DataArgs).prepare_data(data)
    
    eval_loader = data_out[0]

    outputs = TravNet.spike_sorter(ModelArgs, eval_loader).cpu().numpy()
    
    # TODO Output the results in an open source file format (ie neuroshare) for loading 
    # into other software

    return outputs
if __name__ == '__main__':
    fire.Fire(main)



