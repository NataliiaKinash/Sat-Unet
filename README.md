# U-NET on satellite images

Both the training set and the validation set are lists. Each entry in the list is a dictionary with keys “image” and “mask”. Both of these contain numpy arrays with image data.

The image data is captured from the European Space Agency’s (ESAs) Sentinel-2 satellite with a pixel size of 10 m x 10 m, meaning that each image corresponds to 2.56 km x 2.56 km. The three channels correspond to raw channel readings (values in the range from 0 to thousands) from the sensor in Blue-Green-Red order.
The mask is a binary mask with 1s denoting pixels that contain buildings and 0s elsewhere.

The dataset should be placed into the same directory.

I used python 3.8 for this project:

`conda create --name myenv python=3.8`

To install the requirements, run in your virtual environment:

`conda install --yes --file requirements.txt`

To run the model on a dataset, please type:

`python3 cli.py model_path dataset_path`
