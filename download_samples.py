import requests
from tqdm import tqdm

def download_file(url, file_name):
    """
    Downloads a file from the given URL using requests and saves it with the specified file name.
    
    Args:
        url (str): The URL of the file to download.
        file_name (str): The name of the file to save, including the file extension.
    """
    try:
        # Send a HTTP request to the URL of the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception if invalid response
        
        # Get the total size of the file
        total_size = int(response.headers.get('content-length', 0))
        
        # Initialize the progress bar
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        # Open the file in write mode
        with open(file_name, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024):  # Read chunks of 1 KB
                fd.write(chunk)  # Write the chunk to the file

                # Update the progress bar
                progress_bar.update(len(chunk))

        # Close the progress bar
        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong")

        print(f"File '{file_name}' downloaded successfully.")
    except Exception as e:
        print(f"Failed to download the file. Error: {e}")
    
    return None

url_sample = 'https://huggingface.co/datasets/neurodynamics-ai/sample_1chan/resolve/main/sample_1chan.dat'
url_weights = 'https://huggingface.co/neurodynamics-ai/travnetCNN16k/resolve/main/travnetCNN16k.pth'

fn_sample = './/data//sample_1chan.dat'
fn_weights = './/models//travnetcnn16k.pth'

download_file(url_sample, fn_sample)
download_file(url_weights, fn_weights)


