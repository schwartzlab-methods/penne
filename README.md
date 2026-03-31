# PENNE - Phase-to-Expression Neural Network Estimator
Implementation of PENNE, a method for inferring transcriptome from phase-contrast microscopy images.

Read the [paper](www.schwartzlab.ca) and the [documentations](documentations.md)

## Installing PENNE

### Installing using PyPI

PENNE is available on the Python Package Index (PyPI) to be installed with `pip` directly. It is strongly recommended to create a virtual environment before installing, as this is built on ```numpy 1.x```, which means it will probably fail if you try to install it in an environment that is already built with ```numpy 2.x``` due to the massive architecture change for ```numpy```.

To install, run:

```bash
virtualenv --no-download penne
source penne/bin/activate 
pip install pcm-penne
```

### Installing Locally

Alternatively, you may also install SPAGHETTI from the GitHub repository directly. To do that, first create a virtual Python environment and install SPAHETTI locally.

```bash
virtualenv --no-download penne
source penne/bin/activate 
git clone https://github.com/schwartzlab-methods/penne
cd penne
pip install .
```

## Inferences using SPAGHETTI

An example workflow of how to use PENNE to infer gene expression from your phase-contrast microscopy images can be found at `./tutorials/inference.ipynb`. You may supply your own checkpoint files, but if none is supplied, PENNE will automatically download it from the official GitHub repository.

## Inferences with the CLI tool

Alternatively, you can also run inferences using the CLI interface to perform quick inferences. To do this, after you have installed PENNE, run:

```bash
python3 penne --input path_to_directory_with_your_images \
--output path_to_directory_to_save_the_images
```

You can also optionally use ```--penne_checkpoint```, ```--spaghetti_checkpoint```, and ```--gene_names``` if you are not using the default pre-trained model.

## Inferences with Docker

For a dependency-free and reproducible environment, the CLI inference tool of PENNE is available as a Docker image. To use it, ensure you have Docker installed, then run:

### Option 1: Use the Pre-built Image from Docker Hub

The official image is hosted on Docker Hub.

1.  **Pull the latest image:**
    ```bash
    docker pull yinnikun/penne:latest
    ```

2.  **Run Inference:**

    To run inference, you need to mount a local directory into the container. This directory should contain your input images and the model checkpoint. The container will write the output images back to this same directory.

    Let's say your local data is organized as follows:
    ```
    /path/to/your/data/
    ├── inputs/
    │   ├── image1.tif
    │   └── image2.tif
    ├── penne.ckpt <-- optional, if not supplied it will be downloaded
    ├── spaghetti.ckpt <-- optional, if not supplied it will be downloaded
    └── outputs/  <-- This will be created
    ```

    Execute the following command:
    ```bash
    docker run --rm -v "/path/to/your/data:/data" yinnikun/penne:latest \
      --input /data/inputs \
      --penne_checkpoint /data/penne.ckpt \
      --spaghetti_checkpoint /data/penne.ckpt
    ```
    -   `--rm`: Automatically removes the container when it exits.
    -   `-v "/path/to/your/data:/data"`: Mounts your local data directory into the `/data` directory inside the container. **Remember to use absolute paths.**

### Option 2: Build the Image Locally

You can also build the Docker image directly from the `dockerfile` in this repository.

1.  **Build the image:**
    ```bash
    docker build -t penne:latest .
    ```

2.  **Run Inference:**
    The `docker run` command is the same as above, just replace the image name:
    ```bash
    docker run --rm -v "/path/to/your/data:/data" penne:latest \
      --input /data/inputs \
      --output /data/outputs \
      --penne_checkpoint /data/penne.ckpt \
      --spaghetti_checkpoint /data/penne.ckpt
    ```

## Training your own model
You can also train your own model to perform the inferences. See the [documentations](documentations.md) for the details on how to use the ```TrainPenne``` class.
