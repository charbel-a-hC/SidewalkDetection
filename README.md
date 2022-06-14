# Sidwalk Detection
This project is an implementation of image semantic segmentation where a robot detects the sidewalk on a pixel-level precision in order to enhance its navigation.

## Repository Structure
`dataset/train`: training data directory - 276 images with their lables (`dataset/train/image`, `dataset/train/label`). An instance of a dataset; label and raw image is shown below: 
<p align="center">
    <img src="https://user-images.githubusercontent.com/56930805/173705012-57957a9e-78b8-4333-9624-292ba6a1e4f4.png"><br/>
    <em>The highway-v0 environment.</em>
</p>


`dataset/val`: validation data directory - 45 images with their labels. 
`scripts`:
`training`:

## Running Training
To run the training, you can run the following command:
```python
python3 training/experiments/run_training.py
```
## Environment Setup

Clone and `cd` in the repository before running any of the commands:
```bash
git clone https://github.com/charbel-a-hC/SidewalkDetection.git
cd ups-mv-gans
```
You also need to install `python3` locally if you wish to run the notebook on a **local** environment. This automatically install `python3.6.9`. For Ubuntu:
```bash
sudo apt-get install python3.7 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-distutils
```
And you need to update your `pip`:
```bash
/usr/bin/python3 -m pip install --upgrade pip
```
### Docker
If you have docker installed:
```bash
docker build . -t sidewalk-detection
docker run -it --rm -v --runtime=nvidia ${PWD}:/SidewalkDetection
```
### Local Environment (Ubuntu-18.04) - Poetry

Simply run the make command in the repository:
```bash
make env
```
A virtual environment will be created after running the above command. In the same shell, run:
```bash
poetry shell
```
This will activate the environment and you start running any script from this stage.

### Local Environment - Conda
You can download Anaconda [here](https://docs.anaconda.com/anaconda/install/index.html).
After the download, open an anaconda navigator prompt if you're on windows and run the following commands:
```bash
conda env create -f environment.yml
conda activate ml
```
**Note**: If you're on Linux, you can open a normal terminal and run the following command before creating the environment:
```bash
conda activate base
```

### Google Colaboratory
You can open the notebook in Google Colab here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
