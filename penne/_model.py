import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random
import _modules as modules
from _utils import init_spaghetti

#todo: finish the inference class
class Penne():
    def __init__(self, spaghetti_model_path: str):
        """
        Initialize the model
        args:
            spaghetti_model_path: str, the path to the SPAGHETTI model checkpoint
        """
        self.generator = init_spaghetti(spaghetti_model_path)
        # set attributes
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device


class TrainPenne(pl.LightningModule):
    def __init__(self, num_genes, 
                 converter, feature_extractor,
                 end_to_end=False,
                 num_cell_types=0,
                 bio_feature_size=960,
                 domain_feature_size=64,
                 up_marker_genes=None,
                 domain_weight = 5.0,
                 second_order_weight=0.1,
                 cell_type_weight=0.0,
                 marker_gene_weight=0.0,
                 orthogonal_loss_weight=0.0,
                 cosine_weight = 2.0,
                 lr=1e-3,
                 if_ortho=True,
                 convert_for_pcm=True,
                 do_gmlp=True, across_cell=False):
        '''Gene expression prediction model for Visium HD data.

        Args:
            num_genes (int): 
                Number of genes to predict.
            converter (nn.Module): 
                Module to convert image features to gene expression.
            feature_extractor (list(Callable, Callable)): 
                Preprocessor and Model to extract features from images.
            end_to_end (bool, optional): 
                Whether to train the model end-to-end. Defaults to False.
            num_cell_types (int, optional): 
                Number of cell types in the dataset. Defaults to 0.
            bio_feature_size (int, optional): 
                Size of the biological feature representation. Defaults to 960.
            domain_feature_size (int, optional): 
                Size of the domain feature representation. Defaults to 64.
            up_marker_genes (dict[int, list[str]], optional): 
                Mapping from cell type indices to lists of upregulated marker genes. Defaults to None.
            domain_weight (float, optional): 
                Weight for domain adaptation loss. Defaults to 5.0.
            second_order_weight (float, optional): 
                Weight for second-order loss. Defaults to 0.1.
            cell_type_weight (float, optional): 
                Weight for cell type classification loss. Defaults to 0.0.
            marker_gene_weight (float, optional): 
                Weight for marker gene prediction loss. Defaults to 0.0.
            lr (float, optional): 
                Learning rate for the optimizer. Defaults to 1e-3.
            do_gmlp (bool, optional): 
                Whether to use Gated MLP for prediction. Defaults to True.
            across_cell (bool, optional): 
                Whether to use across-cell information. Defaults to False.
            orthogonal_loss_weight (float, optional): 
                Weight for orthogonal loss. Defaults to 0.0.
            cosine_weight (float, optional):
                Weight for cosine similarity loss. Defaults to 2.0.
            if_ortho (bool, optional): 
                Whether to use orthogonal translator. Defaults to True.
            convert_for_pcm (bool, optional): 
                Whether to convert PCM images. Defaults to True.
        '''
        super(TrainPenne, self).__init__()
        # modules
        if if_ortho:
            self.bio_feature_size = bio_feature_size
            self.domain_feature_size = domain_feature_size
            # biology translator
            self.feature_translator = modules.OrthogonalTranslator(feature_in=1024, feature_out=1024)
            # domain classifier
            self.domain_separator = modules.DomainDiscriminator(feature_in=self.domain_feature_size, do_reversal=False)
            self.orthogonal_loss_weight = orthogonal_loss_weight
            # HSIC projector
            self.projector_pcm = modules.HSICProjector(d_x=self.bio_feature_size, d_y=self.domain_feature_size, proj_dim=128)
            self.projector_he = modules.HSICProjector(d_x=self.bio_feature_size, d_y=self.domain_feature_size, proj_dim=128)
        else:
            self.bio_feature_size = 1024
            self.domain_feature_size = 0
        self.domain_discriminator = modules.DomainDiscriminator(self.bio_feature_size, alpha=domain_weight)
        self.cell_type_classifier = modules.CellTypeClassifier(input_size=self.bio_feature_size, hidden_size=512, num_classes=num_cell_types)
        self.cell_type_criterion = nn.CrossEntropyLoss()
        if do_gmlp: # Use Gated MLP for prediction
            self.predictor = modules.PredictorGMLP(input_size=self.bio_feature_size, hidden_size=4056, output_size=num_genes)
        else:
            self.predictor = modules.Predictor(input_size=self.bio_feature_size, hidden_size=4056, output_size=num_genes)
        if up_marker_genes:
            self.up_marker_genes_dict = up_marker_genes
        # feature extractors
        self.image_processor, self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        if converter:
            self.converter = converter
        else:
            self.converter = SpaghettiGenerator(3, 9)
        if not end_to_end:
            for param in self.converter.parameters():
                param.requires_grad = False
            converter.eval()
        # hyperparameters
        self.end_to_end = end_to_end
        self.lr = lr
        self.coral_loss_weight = second_order_weight
        self.cell_type_weight = cell_type_weight
        self.marker_gene_weight = marker_gene_weight
        self.second_stage_training = True if up_marker_genes else False
        self.make_ortho = if_ortho
        self.convert_for_pcm = convert_for_pcm
        self.across_cell = across_cell
        self.cosine_weight = cosine_weight
        # loss functions
        self.criterion = nn.HuberLoss() #MSE + MAE for sparse data
        self.cosine_similarity = nn.CosineEmbeddingLoss()
        self.domain_criterion = nn.BCELoss()
        if if_ortho:
            self.domain_separation_criterion = nn.BCELoss()

        self.save_hyperparameters(ignore=["converter", "feature_extractor"])
    
    @staticmethod
    def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        CORAL = covariance alignment
        Compute CORAL loss between source and target feature maps.
        Use to align the second-order statistics of the source and target distributions.
        Args:
            source (torch.Tensor): Source feature map.
            target (torch.Tensor): Target feature map.
        Returns:
            torch.Tensor: CORAL loss.
        """
        d = source.size(1)  # feature dimension
        ns = source.size(0)  # batch size
        nt = target.size(0)

        # Check if batch sizes are valid
        if ns < 2 or nt < 2:
            return torch.tensor(0.0, device=source.device)

        # Source covariance
        xm = source - source.mean(dim=0, keepdim=True)
        xc = xm.t() @ xm / (ns - 1)

        # Target covariance
        xmt = target - target.mean(dim=0, keepdim=True)
        xct = xmt.t() @ xmt / (nt - 1)

        # Frobenius norm
        loss = torch.mean((xc - xct) ** 2)
        return loss / (4 * d * d)  # normalize by feature dimension
    
    @staticmethod
    def orthogonal_loss(biology: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        '''Compute the orthogonal loss between biology and target feature representations.

        Args:
            biology (torch.Tensor): 
                Biology feature representation of shape (batch_size, num_features).
            domain (torch.Tensor): 
                Domain feature representation of shape (batch_size, num_features).

        Returns:
            torch.Tensor: 
                Orthogonal loss. 
                Computed using the cosine similarity between the biology and domain feature representations.
        '''
        biology = biology - biology.mean(dim=0, keepdim=True)
        domain = domain - domain.mean(dim=0, keepdim=True)

        orthogonality_matrix = torch.matmul(biology.T, domain) / biology.size(0)
        loss = torch.sum(orthogonality_matrix ** 2)
        return loss

    @staticmethod
    def hsic_rbf(x, y, sigma2_x=None, sigma2_y=None):
        """
        Biased HSIC estimate with RBF kernels (works well in practice).
        x: [B, Dx], y: [B, Dy]
        Returns a scalar HSIC >= 0. Larger => more dependence.
        """
        def _rbf_kernel(x, sigma2=None, eps=1e-8):
            # x: [B, D]
            # pairwise squared distances
            d2 = torch.cdist(x, x, p=2.0) ** 2 # [B,B]
            if sigma2 is None:
                # median heuristic (detach to stabilize); add eps to avoid zero
                sigma2 = torch.median(d2.detach())
                sigma2 = torch.clamp(sigma2, min=eps)
            K = torch.exp(-d2 / (2.0 * sigma2))
            return K
        
        B = x.size(0)
        if B < 4:
            # HSIC needs some batch size; return 0 to avoid noise
            return x.new_tensor(0.0)

        # Center features to reduce mean effects (optional but helpful)
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        Kx = _rbf_kernel(x, sigma2_x)
        Ky = _rbf_kernel(y, sigma2_y)

        # Center Gram matrices: H = I - 1/B * 11^T
        H = torch.eye(B, device=x.device) - (1.0 / B) * torch.ones(B, B, device=x.device)
        Kc = H @ Kx @ H
        Lc = H @ Ky @ H

        # Biased HSIC (normalized by (B-1)^2 to keep scales reasonable)
        hsic = (Kc * Lc).sum() / ((B - 1) ** 2)
        return hsic

    def forward(self, x: torch.Tensor, if_convert: bool=False, if_normalize=True, 
                input_feature=False, scramble=False) -> torch.Tensor:
        '''Forward pass for the model.

        Args:
            x (torch.Tensor): 
                Input tensor of the image in the shape of (batch_size, num_channels, height, width).
            if_convert (bool, optional): 
                Whether to use the converter to convert into H&E like images. Defaults to False.
            if_normalize (bool, optional):
                Whether to normalize the output gene expression using log2(CPM + 1). Defaults to True.
            input_feature (bool, optional):
                Whether the input is already a feature representation. Defaults to False.
            scramble (bool, optional):
                Whether to scramble the input image for testing. Defaults to False.

        Returns:
            torch.Tensor: 
                Output tensor of the gene expression inference in the shape of (batch_size, num_genes).
        '''
        # if_convert is used to determine whether to use the converter or not
        torch.manual_seed(42)
        if not input_feature:
            if scramble:
                x = x[:, :, torch.randperm(x.size(2)), :][:, :, :, torch.randperm(x.size(3))]
            if if_convert:
                x = self.converter(x)
            x_converted = self.image_processor(x)      
            x = self.feature_extractor(x_converted).last_hidden_state[:, 0, :].view(x.shape[0], -1)
        # x = self.translator(x)
        if self.make_ortho:
            # if if_convert:
            #     x = self.feature_biology_translator_pcm(x)
            # else:
            x = self.feature_translator(x)[:, :self.bio_feature_size]
        x = self.predictor(x)
        x = torch.clamp(x, min=0)
        if if_normalize:
            x = x / (torch.sum(x, dim=-1, keepdim=True)+1e-10) * 1e6
            x = torch.log2(x + 1)
        return x
    
    def compute_feature(self, x: torch.Tensor, if_convert: bool=False, 
                        if_translate: bool=True, if_ortho: bool=True, scramble=False) -> torch.Tensor:
        '''Compute feature representation for the input tensor.

        Args:
            x (torch.Tensor): 
                Input tensor of the image in the shape of (batch_size, num_channels, height, width).
            if_convert (bool, optional): 
                Whether to use the converter to convert into H&E like images. Defaults to False.
            if_translate (bool, optional): 
                Whether to use the translator for better domain alignment. Defaults to True.
            if_ortho (bool, optional): 
                Whether to use the orthogonal translator. Defaults to True.
            scramble (bool, optional):
                Whether to scramble the input image for testing. Defaults to False.

        Returns:
            torch.Tensor: 
                Feature representation of the input tensor in the shape of (batch_size, num_features).
        '''
        torch.manual_seed(42)
        with torch.no_grad():
            if scramble:
                x = x[:, :, torch.randperm(x.size(2)), :][:, :, :, torch.randperm(x.size(3))]
            if if_convert:
                x = self.converter(x)
            x = self.image_processor(x)
            x = self.feature_extractor(x).last_hidden_state[:, 0, :].view(x.shape[0], -1)
            # if if_translate:
                # x = self.translator(x)
            if if_ortho:
                # if if_convert:
                #     x = self.feature_biology_translator_pcm(x)
                # else:
                x = self.feature_translator(x)
            if self.make_ortho:
                x = x[:, :self.bio_feature_size]
            return x

    def compute_gate(self, x: torch.Tensor, if_convert: bool=False, 
                     if_translate: bool=True, if_ortho: bool=True, 
                     input_feature: bool=False, scramble=False) -> torch.Tensor:
        '''Compute gating vector for the input tensor.

        Args:
            x (torch.Tensor): 
                Input tensor of the image in the shape of (batch_size, num_channels, height, width).
            if_convert (bool, optional): 
                Whether to use the converter to convert into H&E like images. Defaults to False.
            if_translate (bool, optional): 
                Whether to use the translator for better domain alignment. Defaults to True.
            if_ortho (bool, optional): 
                Whether to use the orthogonal translator. Defaults to True.
            scramble (bool, optional):
                Whether to scramble the input image for testing. Defaults to False.
            

        Returns:
            torch.Tensor: 
                Gating vector of the input tensor in the shape of (num_layers, batch_size, num_features).
        '''
        torch.manual_seed(42)
        if not input_feature:
            if scramble:
                x = x[:, :, torch.randperm(x.size(2)), :][:, :, :, torch.randperm(x.size(3))]
            if if_convert:
                x = self.converter(x)
            x = self.image_processor(x)
            x = self.feature_extractor(x).last_hidden_state[:, 0, :].view(x.shape[0], -1)
            # if if_translate:
                # x = self.translator(x)
        if if_ortho:
            # if if_convert:
            #     x = self.feature_biology_translator_pcm(x)
            # else:
            x = self.feature_translator(x)[:, :self.bio_feature_size]
        x = self.predictor.get_gates(x)
        return x

    def compute_domain_feature(self, x: torch.Tensor, if_convert: bool=False, 
                                 if_translate: bool=True, if_ortho: bool=True, 
                                 input_feature: bool=False, scramble=False) -> torch.Tensor:
        '''Compute domain feature representation for the input tensor.

        Args:
            x (torch.Tensor): 
                Input tensor of the image in the shape of (batch_size, num_channels, height, width).
            if_convert (bool, optional): 
                Whether to use the converter to convert into H&E like images. Defaults to False.
            if_translate (bool, optional): 
                Whether to use the translator for better domain alignment. Defaults to True.
            if_ortho (bool, optional): 
                Whether to use the orthogonal translator. Defaults to True.
            scramble (bool, optional):
                Whether to scramble the input image for testing. Defaults to False.

        Returns:
            torch.Tensor: 
                Domain feature representation of the input tensor in the shape of (batch_size, num_features).
        '''
        torch.manual_seed(42)
        if not input_feature:
            if scramble:
                x = x[:, :, torch.randperm(x.size(2)), :][:, :, :, torch.randperm(x.size(3))]
            if if_convert:
                x = self.converter(x)
            x = self.image_processor(x)
            x = self.feature_extractor(x).last_hidden_state[:, 0, :].view(x.shape[0], -1)
        # if if_translate:
            # x = self.translator(x)
        if if_ortho:
            # if if_convert:
            #     x = self.feature_domain_translator_pcm(x)
            # else:
            x = self.feature_translator(x)[:, self.bio_feature_size:]
        return x

    def _marker_margin_loss(self, pred_expr: torch.Tensor, cell_types: torch.Tensor,
                            marker_dict: dict, margin: float=1.0,
                            across_cell: bool=False) -> torch.Tensor:
        '''Compute the marker margin loss.

        Args:
            pred_expr (torch.Tensor): 
                Predicted gene expression values in the shape of (batch_size, num_genes).
            cell_types (torch.Tensor): 
                Cell type labels in the shape of (batch_size,).
            marker_dict (dict[Int, List[Int]]): 
                Dictionary mapping cell type indices to marker gene indices in exp matrix.
            margin (float, optional): 
                Margin for the loss. Defaults to 1.0.
            across_cell (bool, optional): 
                Whether to compute the loss across cell types instead of within cell types. Defaults to False.

        Returns:
            torch.Tensor: Computed loss value of the marker margin loss.
        '''
        loss = torch.tensor(0.0, device=pred_expr.device)
        if across_cell:
            # Compare the marker gene exp across all cells. 
            # The marker genes should be more expressed in the target cell type than in any other cell type
            unique_cell_types = torch.unique(cell_types)
            for cell_type in unique_cell_types:
                marker_genes = torch.tensor(marker_dict[cell_type.item()])
                num_marker_genes = marker_genes.sum().item()
                if num_marker_genes == 0:
                    continue
                # get the marker gene exp for this cell type and other cell types
                marker_genes = marker_genes.bool().to(self.device)
                current_cell_mask = (cell_types == cell_type).to(self.device)
                pred_expr_current_cell = pred_expr[current_cell_mask, :]
                pred_expr_other_cells = pred_expr[~current_cell_mask, :]
                if (pred_expr_current_cell.size(0) == 0) or (pred_expr_other_cells.size(0) == 0):
                    continue
                # get the mean marker gene exp for each gene
                marker_genes = marker_genes.to(self.device)
                marker_expr_current_cell = pred_expr_current_cell[:, marker_genes].mean(dim=0, keepdim=True).view(1, -1)
                marker_expr_other_cells = pred_expr_other_cells[:, marker_genes].mean(dim=0, keepdim=True).view(1, -1)
                # calculate loss
                loss += F.margin_ranking_loss(
                    marker_expr_current_cell, marker_expr_other_cells,
                    target=torch.ones_like(marker_expr_current_cell), margin=margin
                )
            return loss / unique_cell_types.size(0) 
        else: # Compare the marker gene exp within each cell type
            for i in range(pred_expr.size(0)):
                cell_type = cell_types[i].item()
                marker_genes = torch.tensor(marker_dict[cell_type])
                num_marker_genes = marker_genes.sum().item()
                if num_marker_genes == 0:
                    continue
                non_marker_genes = 1 - marker_genes
                # Sample a few non-marker genes to compare against
                idx_non_marker = non_marker_genes.nonzero(as_tuple=False).flatten()
                sampled_non = random.sample(range(idx_non_marker.shape[0]), int(num_marker_genes))
                sampled_non_marker_genes_idx = idx_non_marker[sampled_non]
                marker_vals = pred_expr[i, marker_genes.bool().to(self.device)].view(1,-1)
                non_marker_vals = pred_expr[i, sampled_non_marker_genes_idx].view(1,-1)
                # calculate loss
                loss += F.margin_ranking_loss(
                    marker_vals, non_marker_vals, target=torch.ones_like(marker_vals), margin=margin
                )
            return loss / pred_expr.size(0)

    def training_step(self, batch: tuple, batch_idx: int):
        '''Perform a single training step.

        Args:
            batch (tuple): A tuple containing the input data for the batch. This contains the following elements:
                - he_image (torch.Tensor): The HE image tensor in the shape of (batch_size, channels, height, width).
                - mtx (torch.Tensor): The gene expression matrix.
                - pcm_image (torch.Tensor): The PCM image tensor.
                - _ (str): HE image path, not used during training
                - cell_type (torch.Tensor): The cell type labels in the shape of (batch_size,).
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The computed loss for the batch.
        '''
        he_image, mtx, pcm_image, _, cell_type = batch
        # obtain the features
        if self.convert_for_pcm:
            pcm_converted = self.converter(pcm_image)
        else:
            pcm_converted = pcm_image
        pcm_converted = self.image_processor(pcm_converted)
        he_converted = self.image_processor(he_image)
        pcm_translated = self.feature_extractor(pcm_converted).last_hidden_state[:, 0, :].view(pcm_image.shape[0], -1).requires_grad_()#.detach()
        he_translated = self.feature_extractor(he_converted).last_hidden_state[:, 0, :].view(he_image.shape[0], -1).requires_grad_()#.detach()
        if hasattr(self, "feature_translator"):
            he_translated_biology = self.feature_translator(he_translated)[:, :self.bio_feature_size]
            pcm_translated_biology = self.feature_translator(pcm_translated)[:, :self.bio_feature_size]
            he_translated_domain = self.feature_translator(he_translated)[:, self.bio_feature_size:]
            pcm_translated_domain = self.feature_translator(pcm_translated)[:, self.bio_feature_size:]
            # loss to enforce orthogonality between biological and domain features
            pcm_bio_proj, pcm_domain_proj = self.projector_pcm(pcm_translated_biology, pcm_translated_domain)
            he_bio_proj, he_domain_proj = self.projector_he(he_translated_biology, he_translated_domain)
            dot_product_loss = (self.orthogonal_loss(pcm_bio_proj, pcm_domain_proj) 
                                + self.orthogonal_loss(he_bio_proj, he_domain_proj)) / 2

            hsic_loss = (self.hsic_rbf(pcm_bio_proj, pcm_domain_proj) 
                        + self.hsic_rbf(he_bio_proj, he_domain_proj)) / 2
            ortho_loss_non_weight = dot_product_loss + hsic_loss
            ortho_loss = self.orthogonal_loss_weight * ortho_loss_non_weight
        else:
            he_translated_biology = he_translated
            pcm_translated_biology = pcm_translated
            he_translated_domain = he_translated
            pcm_translated_domain = pcm_translated

        # Domain alignment
        # this part is for domain adaptation, uses biology features only
        # DANN
        pred_discriminator_fake = self.domain_discriminator(he_translated_biology)
        pred_discriminator_real = self.domain_discriminator(pcm_translated_biology)
        fake_labels = torch.zeros_like(pred_discriminator_fake)
        real_labels = torch.ones_like(pred_discriminator_real)
        domain_loss = (self.domain_criterion(pred_discriminator_fake, fake_labels) + 
                       self.domain_criterion(pred_discriminator_real, real_labels)) / 2
        # Coral Loss
        coral_loss = self.coral_loss(he_translated_biology, pcm_translated_biology)
        # cell type loss
        cell_type_pred = self.cell_type_classifier(pcm_translated_biology)
        cell_type_loss = self.cell_type_criterion(cell_type_pred, cell_type.to(self.device))

        # this part is for domain seperation
        he_domain_separated = self.domain_separator(he_translated_domain)
        pcm_domain_separated = self.domain_separator(pcm_translated_domain)
        he_domain_labels = torch.zeros_like(he_domain_separated)
        pcm_domain_labels = torch.ones_like(pcm_domain_separated)
        domain_separation_loss_non_weight = (self.domain_separation_criterion(he_domain_separated, he_domain_labels) + 
                                            self.domain_separation_criterion(pcm_domain_separated, pcm_domain_labels)) / 2

        domain_separation_loss = self.orthogonal_loss_weight * domain_separation_loss_non_weight


        # Predictor part
        # This part leverages the biological translated features
        exp_pred = self.predictor(he_translated_biology)
        # normalize to counts per million and log2 transform
        exp_pred = torch.clamp(exp_pred, min=0)
        exp_pred = exp_pred / (torch.sum(exp_pred, dim=-1, keepdim=True)+1e-10) * 1e6
        exp_pred = torch.log2(exp_pred + 1)
        prediction_loss = self.criterion(exp_pred, mtx.to(self.device))
        cosine_target = torch.ones(exp_pred.size(0)).to(self.device)
        cosine_pred_loss = self.cosine_similarity(exp_pred, mtx.to(self.device), cosine_target)

        if self.second_stage_training:
            pred_exp_pcm = self.predictor(pcm_translated_biology)
            # normalize to counts per million and log2 transform
            pred_exp_pcm = torch.clamp(pred_exp_pcm, min=0)
            pred_exp_pcm = pred_exp_pcm / (torch.sum(pred_exp_pcm, dim=-1, keepdim=True)+1e-10) * 1e6
            pred_exp_pcm = torch.log2(pred_exp_pcm + 1)
            # marker gene loss
            marker_gene_loss = self._marker_margin_loss(pred_exp_pcm, cell_type, self.up_marker_genes_dict, across_cell=self.across_cell)
            coral_pcm = self.coral_loss(pred_exp_pcm, exp_pred)
            coral_loss = coral_loss + coral_pcm

            # total loss for training
            total_loss = prediction_loss + self.cosine_weight * cosine_pred_loss + domain_loss + self.coral_loss_weight * coral_loss + self.cell_type_weight * cell_type_loss + self.marker_gene_weight *  marker_gene_loss
            metrics = {"train_loss": total_loss.item(), "train_discriminator_loss": domain_loss.item(),
                        "train_coral_loss": coral_loss.item(), "train_cosine_loss": cosine_pred_loss.item(),
                        "train_prediction_loss": prediction_loss.item(), "train_cell_type_loss": cell_type_loss.item(), 
                        "train_marker_gene_loss": marker_gene_loss}
        else:
            # total loss for training
            total_loss = prediction_loss + self.cosine_weight * cosine_pred_loss + domain_loss + self.coral_loss_weight * coral_loss + self.cell_type_weight * cell_type_loss
            metrics = {"train_loss": total_loss.item(), "train_discriminator_loss": domain_loss.item(),
                        "train_coral_loss": coral_loss.item(), "train_cosine_loss": cosine_pred_loss.item(),
                        "train_prediction_loss": prediction_loss.item(), "train_cell_type_loss": cell_type_loss.item()}
        if self.make_ortho:
            total_loss += (ortho_loss + domain_separation_loss)
            metrics["train_ortho_loss"] = ortho_loss_non_weight.item()
            metrics["train_domain_separation_loss"] = domain_separation_loss_non_weight.item()

        self.log_dict(metrics,prog_bar=True)
        return total_loss
 
    def validation_step(self, batch, batch_idx):
        '''Perform a single validation step.

        Args:
            batch (tuple): A tuple containing the input data for the batch. This contains the following elements:
                - he_image (torch.Tensor): The HE image tensor in the shape of (batch_size, channels, height, width).
                - mtx (torch.Tensor): The gene expression matrix.
                - pcm_image (torch.Tensor): The PCM image tensor.
                - _ (str): HE image path, not used during validation
                - cell_type (torch.Tensor): The cell type labels in the shape of (batch_size,).
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The computed loss for the batch.
        '''
        he_image, mtx, pcm_image, name, cell_type = batch
        with torch.no_grad():
            # obtain the features
            if self.convert_for_pcm:
                pcm_converted = self.converter(pcm_image)
            else:
                pcm_converted = pcm_image
            pcm_converted = self.image_processor(pcm_converted)
            he_converted = self.image_processor(he_image)
            pcm_translated = self.feature_extractor(pcm_converted).last_hidden_state[:, 0, :].view(pcm_image.shape[0], -1).detach()
            he_translated = self.feature_extractor(he_converted).last_hidden_state[:, 0, :].view(he_image.shape[0], -1).detach()
            # Translator part
            if hasattr(self, "feature_translator"):
                he_translated_biology = self.feature_translator(he_translated)[:, :self.bio_feature_size]
                pcm_translated_biology = self.feature_translator(pcm_translated)[:, :self.bio_feature_size]
                he_translated_domain = self.feature_translator(he_translated)[:, self.bio_feature_size:]
                pcm_translated_domain = self.feature_translator(pcm_translated)[:, self.bio_feature_size:]
                # loss to enforce orthogonality between biological and domain features
                pcm_bio_proj, pcm_domain_proj = self.projector_pcm(pcm_translated_biology, pcm_translated_domain)
                he_bio_proj, he_domain_proj = self.projector_he(he_translated_biology, he_translated_domain)
                dot_product_loss = (self.orthogonal_loss(pcm_bio_proj, pcm_domain_proj) 
                                    + self.orthogonal_loss(he_bio_proj, he_domain_proj)) / 2

                hsic_loss = (self.hsic_rbf((pcm_bio_proj), pcm_domain_proj) 
                            + self.hsic_rbf((he_bio_proj), he_domain_proj)) / 2
                ortho_loss_non_weight = dot_product_loss + hsic_loss
                ortho_loss = self.orthogonal_loss_weight * ortho_loss_non_weight
            else:
                he_translated_biology = he_translated
                pcm_translated_biology = pcm_translated
                he_translated_domain = he_translated
                pcm_translated_domain = pcm_translated
            # DANN part
            # this part is for domain adaptation, uses biology features only
            pred_discriminator_fake = self.domain_discriminator(he_translated_biology)
            pred_discriminator_real = self.domain_discriminator(pcm_translated_biology)
            fake_labels = torch.zeros_like(pred_discriminator_fake)
            real_labels = torch.ones_like(pred_discriminator_real)
            domain_loss = (self.domain_criterion(pred_discriminator_fake, fake_labels) + 
                        self.domain_criterion(pred_discriminator_real, real_labels)) / 2
            coral_loss = self.coral_loss(he_translated_biology, pcm_translated_biology)
            # PCM Cell Type loss
            cell_type_pred = self.cell_type_classifier(pcm_translated_biology)
            cell_type_loss = self.cell_type_criterion(cell_type_pred, cell_type.to(self.device))

            # this part is for domain seperation
            he_domain_separated = self.domain_separator(he_translated_domain)
            pcm_domain_separated = self.domain_separator(pcm_translated_domain)
            he_domain_labels = torch.zeros_like(he_domain_separated)
            pcm_domain_labels = torch.ones_like(pcm_domain_separated)
            domain_separation_loss_non_weight = (self.domain_separation_criterion(he_domain_separated, he_domain_labels) + 
                                            self.domain_separation_criterion(pcm_domain_separated, pcm_domain_labels)) / 2

            domain_separation_loss = self.orthogonal_loss_weight * domain_separation_loss_non_weight

            # Predictor part
            exp_pred = self.predictor(he_translated_biology)
            exp_pred = torch.clamp(exp_pred, min=0)
            exp_pred = exp_pred / (torch.sum(exp_pred, dim=-1, keepdim=True)+1e-10) * 1e6
            exp_pred = torch.log2(exp_pred + 1)
            prediction_loss = self.criterion(exp_pred, mtx.to(self.device))
            cosine_target = torch.ones(exp_pred.size(0)).to(self.device)
            cosine_pred_loss = self.cosine_similarity(exp_pred, mtx.to(self.device), cosine_target)
            if self.second_stage_training:
                pred_exp_pcm = self.predictor(pcm_translated_biology)
                pred_exp_pcm = torch.clamp(pred_exp_pcm, min=0)
                pred_exp_pcm = pred_exp_pcm / (torch.sum(pred_exp_pcm, dim=-1, keepdim=True)+1e-10) * 1e6
                pred_exp_pcm = torch.log2(pred_exp_pcm + 1)
                # marker gene loss
                marker_gene_loss = self._marker_margin_loss(pred_exp_pcm, cell_type, self.up_marker_genes_dict, across_cell=self.across_cell)
                coral_pcm = self.coral_loss(pred_exp_pcm, exp_pred)
                coral_loss = coral_loss + coral_pcm
                # total loss for validation
                total_loss = prediction_loss + self.cosine_weight * cosine_pred_loss + domain_loss + self.coral_loss_weight * coral_loss + self.cell_type_weight * cell_type_loss + self.marker_gene_weight * marker_gene_loss
                metrics = {"val_loss": total_loss.item(), "val_discriminator_loss": domain_loss.item(),
                           "val_coral_loss": coral_loss.item(), "val_cosine_loss": cosine_pred_loss.item(),
                            "val_prediction_loss": prediction_loss.item(), "val_cell_type_loss": cell_type_loss.item(),
                            "val_marker_gene_loss": marker_gene_loss}
            else:
                # total loss for validation
                total_loss = prediction_loss + self.cosine_weight * cosine_pred_loss + domain_loss + self.coral_loss_weight * coral_loss + self.cell_type_weight * cell_type_loss
                metrics = {"val_loss": total_loss.item(), "val_discriminator_loss": domain_loss.item(), "val_cosine_loss": cosine_pred_loss.item(),
                            "val_coral_loss": coral_loss.item(), "val_prediction_loss": prediction_loss.item(), "val_cell_type_loss": cell_type_loss.item()}
            if self.make_ortho:
                total_loss += (ortho_loss + domain_separation_loss)
                metrics["val_ortho_loss"] = ortho_loss_non_weight.item()
                metrics["val_domain_separation_loss"] = domain_separation_loss_non_weight.item()
            self.log_dict(metrics,prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]