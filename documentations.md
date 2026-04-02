# PENNE - Phase-to-Expression Neural Network Estimator Documentation

Documentation of classes and methods for PENNE. For installing details, see [README.md](README.md). For a tutorial on sample usage, see the directory at ```tutorials/```.

This package provides a PyTorch implementation for inferring gene expression from phase-contrast microscopy images.

---

## 1. `_modules.py`

This file contains the core PyTorch neural network modules, including domain adaptation and prediction components.

### `GradReverse` (torch.autograd.Function)
Gradient reversal layer for adversarial training.
* **`forward(ctx, x, alpha=1.0)`**
    * **Inputs:**
        * `ctx`: Context object used to store information for the backward pass.
        * `x` (Tensor): Input tensor of any shape.
        * `alpha` (float): Scaling factor for gradient reversal.
    * **Outputs:**
        * `x` (Tensor): Output tensor, same shape as input.
* **`backward(ctx, grad_output)`**
    * **Inputs:**
        * `ctx`: Context object from the forward pass.
        * `grad_output`: Gradient of the loss with respect to the output.
    * **Outputs:**
        * (Tensor, None): Gradient of the loss with respect to the input (scaled by `-alpha`), and `None` for the alpha parameter.

### `DomainDiscriminator` (nn.Module)
Classifies feature vectors into one of two domains.
* **`__init__(feature_in=1024, alpha=1.0, do_reversal=True)`**
    * **Inputs:**
        * `feature_in` (int): The input feature dimension.
        * `alpha` (float): The scaling factor for gradient reversal.
        * `do_reversal` (bool): Whether to apply gradient reversal.
* **`forward(x)`**
    * **Inputs:**
        * `x` (Tensor): Input tensor of shape `(batch_size, feature_in)`.
    * **Outputs:**
        * (Tensor): Output tensor of shape `(batch_size, 1)` representing domain probabilities between 0 and 1.

### `OrthogonalTranslator` (nn.Module)
Translator module for domain adaptation of Phase-Contrast Microscopy (PCM) and H&E features.
* **`__init__(feature_in=1024, feature_out=1024)`**
    * **Inputs:**
        * `feature_in` (int): Input feature dimension.
        * `feature_out` (int): Output feature dimension.
* **`forward(x)`**
    * **Inputs:**
        * `x` (Tensor): Input tensor.
    * **Outputs:**
        * (Tensor): Translated output tensor.

### `HSICProjector` (nn.Module)
Projects two different feature spaces into a common space of the same dimension to stabilize HSIC and cosine similarity losses.
* **`__init__(d_x, d_y, proj_dim=128)`**
    * **Inputs:**
        * `d_x` (int): Dimension of the first feature space.
        * `d_y` (int): Dimension of the second feature space.
        * `proj_dim` (int): Dimension of the common projected space.
* **`forward(x, y)`**
    * **Inputs:**
        * `x` (Tensor): First feature space tensor.
        * `y` (Tensor): Second feature space tensor.
    * **Outputs:**
        * `(Tensor, Tensor)`: L2-normalized projected tensors `x` and `y`.

### `GatedMLPBlock` (nn.Module)
Gated MLP block for gene expression prediction.
* **`__init__(input_size, hidden_size, p=0.2)`**
    * **Inputs:**
        * `input_size` (int): Size of the input feature vector.
        * `hidden_size` (int): Size of the hidden layer.
        * `p` (float): Dropout probability.
* **`forward(x)`**
    * **Inputs:**
        * `x` (Tensor): Input tensor of shape `(batch_size, input_size)`.
    * **Outputs:**
        * (Tensor): Output tensor of shape `(batch_size, input_size)`.
* **`get_x_and_gate(x)`**
    * **Inputs:**
        * `x` (Tensor): Input tensor of shape `(batch_size, input_size)`.
    * **Outputs:**
        * `(Tensor, Tensor)`: The module output and the gate tensor, both of shape `(batch_size, input_size)`.

### `PredictorGMLP` (nn.Module)
Predicts the whole transcriptome from image features using a Gated MLP.
* **`__init__(input_size, hidden_size, output_size, num_layers=3, dropout=0.2)`**
    * **Inputs:**
        * `input_size` (int): Size of the input feature vector.
        * `hidden_size` (int): Size of the hidden layers.
        * `output_size` (int): Size of the output gene expression vector.
        * `num_layers` (int): Number of Gated MLP blocks.
        * `dropout` (float): Dropout probability.
* **`forward(x)`**
    * **Inputs:**
        * `x` (Tensor): Input tensor of shape `(batch_size, input_size)`.
    * **Outputs:**
        * (Tensor): Predicted gene expression tensor of shape `(batch_size, output_size)`.
* **`get_gates(x)`**
    * **Inputs:**
        * `x` (Tensor): Input tensor.
    * **Outputs:**
        * (Tensor): Stacked gate values of shape `(num_layers, batch_size, input_size)`.

### `ResidualBlock` (nn.Module)
Backbone of the SPAGHETTI generator network.
* **`__init__(in_channels)`**
    * **Inputs:**
        * `in_channels` (int): Number of input/output channels.
* **`forward(x)`**
    * **Inputs:**
        * `x` (Tensor): Input tensor.
    * **Outputs:**
        * (Tensor): Output tensor after applying residual connections.

### `SpaghettiGenerator` (nn.Module)
Generator network for image-to-image translation.
* **`__init__(in_channels, num_residual_blocks=9)`**
    * **Inputs:**
        * `in_channels` (int): Number of input/output channels (e.g., 3 for RGB).
        * `num_residual_blocks` (int): Number of residual blocks.
* **`forward(x)`**
    * **Inputs:**
        * `x` (Tensor): Image tensor of shape `(batch_size, 3, 256, 256)`.
    * **Outputs:**
        * (Tensor): Normalized, translated image tensor.

### `CellTypeClassifier` (nn.Module)
Classifies cell types from gene expression features.
* **`__init__(input_size, num_classes, hidden_size=512)`**
    * **Inputs:**
        * `input_size` (int): Input feature vector size.
        * `num_classes` (int): Number of cell type classes.
        * `hidden_size` (int): Hidden layer size.
* **`forward(x)`**
    * **Inputs:**
        * `x` (Tensor): Input tensor of shape `(batch_size, input_size)`.
    * **Outputs:**
        * (Tensor): Raw class scores of shape `(batch_size, num_classes)`.

---

## 2. `_utils.py`

Utility functions for initialization, preprocessing, and asset management.

### `init_spaghetti`
Initializes the SPAGHETTI model for image translation.
* **Inputs:**
    * `model_path` (str): Path to the saved model checkpoint.
* **Outputs:**
    * (torch.nn.Module): The loaded `SpaghettiGenerator` network in evaluation mode.

### `pre_processing_phikon`
Sets up the image preprocessing pipeline.
* **Inputs:**
    * `model` (str, optional): HuggingFace model name. Defaults to `None`.
* **Outputs:**
    * (Callable): A function that takes an image batch and returns preprocessed image tensors.

### `download_asset_from_github`
Downloads required assets from the PENNE GitHub repository.
* **Inputs:**
    * `asset_name` (str): The name of the asset (e.g., "spaghetti.ckpt").
    * `save_path` (str, optional): Local directory to save the asset. Defaults to `"~/.cache/penne/"`.
* **Outputs:**
    * `None`

---

## 3. `dataset.py`

PyTorch Dataset implementations for training and inference.

### `InferenceDataset` (torch.utils.data.Dataset)
Dataset handler for generating predictions from input images.
* **`__init__(paths: Union[list[str], Tuple[str], str])`**
    * **Inputs:**
        * `paths`: Directory path(s) containing `.png`, `.jpg`, `.jpeg`, or `.tiff` images.
* **`__len__()`**
    * **Outputs:**
        * (int): Number of valid image files found.
* **`__getitem__(idx: int)`**
    * **Inputs:**
        * `idx` (int): Item index.
    * **Outputs:**
        * `(torch.Tensor, str)`: Preprocessed image tensor and the original filename.

### `TrainingDataset` (torch.utils.data.Dataset)
Dataset handler for paired Visium and LIVECell data.
* **`__init__(tissue_dir: str, mtx_dir: str, livecell_dir: str, use_mtx: bool = True)`**
    * **Inputs:**
        * `tissue_dir` (str): Path to tissue image directory.
        * `mtx_dir` (str): Path to gene expression matrix directory.
        * `livecell_dir` (str): Path to LIVECell image directory.
        * `use_mtx` (bool): Whether to load matrices (if `False`, returns zeros).
* **`__len__()`**
    * **Outputs:**
        * (int): Number of items based on tissue images.
* **`__getitem__(idx: int)`**
    * **Inputs:**
        * `idx` (int): Item index.
    * **Outputs:**
        * `(torch.Tensor, torch.Tensor, torch.Tensor, str, int)`: Contains the H&E image tensor, expression matrix tensor, LIVECell image tensor, H&E file path, and LIVECell class index.
* **`_find_all_files(path)`** (Static Method)
    * **Inputs:** `path` (str)
    * **Outputs:** (list[str]) All file paths within the directory tree.
* **`_write_attributes(livecell_dir)`**
    * **Inputs:** `livecell_dir` (list[str])
    * **Outputs:** `None`. Mutates instance attributes for dataset mapping.

---

## 4. `model.py`

Contains the PyTorch Lightning modules handling the model architecture, training loops, and inference.

### `Penne` (pl.LightningModule)
The primary inference wrapper for the trained PENNE model.
* **`__init__(...)`**
    * **Inputs:**
        * `spaghetti_model_path`, `penne_model_path`, `gene_names`, `high_confidence_genes` (str/Path/None): Asset paths.
        * `num_genes` (int): Default 18085.
        * `do_high_confidence_genes` (bool): Default False.
        * `feature_extractor` (tuple, optional): Processor and extractor objects.
        * `bio_feature_size` (int): Default 960.
        * `domain_feature_size` (int): Default 64.
* **`forward(loader: DataLoader, save_path: Union[str, None] = None)`**
    * **Inputs:**
        * `loader` (DataLoader): Batch provider for preprocessed PCM images.
        * `save_path` (str, optional): Directory to save output `.h5ad` file.
    * **Outputs:**
        * (`ad.AnnData`): AnnData object containing the predicted gene expressions.

### `TrainPenne` (pl.LightningModule)
The core model used for training.
* **`__init__(...)`**
    * **Inputs:** Highly parameterized initialization taking parameters like `num_genes`, `converter`, `feature_extractor`, `end_to_end`, `num_cell_types`, various layer weights (`domain_weight`, `cosine_weight`, etc.), and architecture toggles (`do_gmlp`, `if_ortho`, `convert_for_pcm`).
* **`coral_loss(source: torch.Tensor, target: torch.Tensor)`** (Static Method)
    * **Inputs:**
        * `source` (torch.Tensor): Source feature map.
        * `target` (torch.Tensor): Target feature map.
    * **Outputs:**
        * (torch.Tensor): Calculated CORAL covariance alignment loss.
* **`orthogonal_loss(biology: torch.Tensor, domain: torch.Tensor)`** (Static Method)
    * **Inputs:**
        * `biology` (torch.Tensor): Biology features `(batch_size, num_features)`.
        * `domain` (torch.Tensor): Domain features `(batch_size, num_features)`.
    * **Outputs:**
        * (torch.Tensor): Calculated orthogonal loss.
* **`hsic_rbf(x, y, sigma2_x=None, sigma2_y=None)`** (Static Method)
    * **Inputs:** Tensors `x`, `y` and optional kernel variance scalars.
    * **Outputs:** (torch.Tensor): Biased HSIC scalar estimate.
* **`forward(x: torch.Tensor, if_convert=False, if_normalize=True, input_feature=False, scramble=False)`**
    * **Inputs:**
        * `x` (torch.Tensor): Image tensor or feature tensor.
        * `if_convert`, `if_normalize`, `input_feature`, `scramble` (bool): Toggles for processing steps.
    * **Outputs:**
        * (torch.Tensor): Output gene expression tensor `(batch_size, num_genes)`.
* **`compute_feature(...)` / `compute_domain_feature(...)`**
    * **Inputs:** Tensors and boolean toggles identical to `forward`.
    * **Outputs:** (torch.Tensor): Biological or Domain feature representations respectively.
* **`compute_gate(x: torch.Tensor, ...)`**
    * **Inputs:** Image tensor and boolean processing toggles.
    * **Outputs:** (torch.Tensor): Gating vector `(num_layers, batch_size, num_features)`.
* **`_marker_margin_loss(pred_expr, cell_types, marker_dict, margin=1.0, across_cell=False)`**
    * **Inputs:**
        * `pred_expr` (torch.Tensor): Predictions.
        * `cell_types` (torch.Tensor): Labels.
        * `marker_dict` (dict): Marker mappings.
        * `margin` (float): Margin value.
        * `across_cell` (bool): Toggle cross-cell computation.
    * **Outputs:**
        * (torch.Tensor): Margin loss scalar.
* **`training_step(batch: tuple, batch_idx: int)`**
    * **Inputs:** Batch payload from `TrainingDataset`.
    * **Outputs:** (torch.Tensor): Total batch loss.
* **`validation_step(batch, batch_idx)`**
    * **Inputs:** Batch payload.
    * **Outputs:** `None`. (Logs metrics implicitly).
* **`configure_optimizers()`**
    * **Outputs:** `([optimizer], [scheduler])` lists.

---

## 5. `cli_inference.py`

Command Line Interface logic for running batch predictions.
* **`main()`**
    * **Inputs:** None directly. (Reads `sys.argv` flags: `--input_dir`, `--output_dir`, `--gene_names`, `--penne_ckpt`, `--spaghetti_ckpt`).
    * **Outputs:** None directly. Evaluates the dataset and saves an `.h5ad` result file to the output directory.