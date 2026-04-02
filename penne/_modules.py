'''
Modules for PENNE - Phase-to-Expression Neural Network Estimator
'''
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
import torch


########! Domain adaptation modules
########! These translate the feature vectors from both the real H&E and synthetic H&E images 
########! into the same space.
########! It uses a Domain Adversarial Neural Network (DANN) approach to align the feature distributions
########! of the two domains.

#* Gradient reversal layer (for adversarial training)
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        '''
        Forward pass for gradient reversal layer
        Args:
            ctx: context, used to store information for backward pass
            x: input tensor of any shape
            alpha: scaling factor for gradient reversal. Defaults to 1.0.
        Returns:
            x: output tensor, same shape as input
        '''
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Backward pass for gradient reversal layer
        Args:
            ctx: context, used to retrieve information from forward pass
            grad_output: gradient of the loss with respect to the output

        Returns:
            gradient of the loss with respect to the input, scaled by -alpha
        '''
        return grad_output.neg() * ctx.alpha, None

#* Discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self, feature_in=1024, alpha=1.0, do_reversal=True):
        '''General purpose discriminator that classify the feature vectors into one of two domains
        Args:
            feature_in (int, optional): The input feature dimension. Defaults to 1024.
            alpha (float, optional): The scaling factor for gradient reversal layer. Defaults to 1.0.
            do_reversal (bool, optional): Whether to apply gradient reversal. Set to True for DANN, false otherwise for a standard discriminator. 
                                        Defaults to True.
        '''
        super(DomainDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_in, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.alpha = alpha
        self.do_reversal = do_reversal

    def forward(self, x):
        '''Forward pass for discriminator

        Args:
            x (Tensor): Input tensor of shape (batch_size, feature_dim)

        Returns:
            Tensor: Output tensor of shape (batch_size, 1) representing domain probabilities. Each value is between 0 and 1.
        Args:
            x (Tensor): Input tensor of shape (batch_size, feature_in)
        '''
        if self.do_reversal:
            out = GradReverse.apply(x, self.alpha)
            return self.model(out)
        else:
            return self.model(x)

#* Orthogonality translator
class OrthogonalTranslator(nn.Module):
    def __init__(self, feature_in=1024, feature_out=1024):
        '''
        Translator module for domain adaptation of PCM features and H&E features

        Args:
            feature_in (int, optional): The input feature dimension. Defaults to 1024.
            feature_out (int, optional): The output feature dimension. Defaults to 1024.
        '''
        super(OrthogonalTranslator, self).__init__()
        self.fc1 = nn.Linear(feature_in, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, feature_out)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.ln1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.ln2(x))
        x = self.dropout(x)
        return self.fc3(x)

#! Prediction modules
#! Predict the whole transcriptome from the image

class HSICProjector(nn.Module):
    def __init__(self, d_x, d_y, proj_dim=128):
        '''
        HSIC Projector module to project two different feature spaces into a common space of the same dimension.
        This stablizes the training when using HSIC and cosine similarity losses for domain adaptation
        Args:
            d_x (int): Dimension of the first feature space.
            d_y (int): Dimension of the second feature space.
            proj_dim (int, optional): Dimension of the common projected space. Defaults to 128.
        '''
        super().__init__()
        self.px = nn.Linear(d_x, proj_dim)
        self.py = nn.Linear(d_y, proj_dim)

    def forward(self, x, y):
        # L2-normalize to stabilize distances
        x = F.normalize(self.px(x), dim=1)
        y = F.normalize(self.py(y), dim=1)
        return x, y

class GatedMLPBlock(nn.Module):
    def __init__(self, input_size, hidden_size, p=0.2):
        '''
        Block of Gated MLP for gene expression prediction.
        Args:
            input_size (int): Size of the input feature vector.
            hidden_size (int): Size of the hidden layer.
            p (float, optional): Dropout probability. Defaults to 0.2.
        '''
        super(GatedMLPBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(p) # Dropout layer to prevent overfitting

    def forward(self, x):
        '''
        Forward pass for Gated MLP block, governed by a gating mechanism
        Gating mechanism allows the model to adaptively control the information flow, 
        enhancing its ability to capture complex yet sparse relationships.
        The gating mechanism is defined as:
            gate = sigmoid(W2 * GELU(W1 * x))
        The output is computed as:
            output = LayerNorm(x + gate * f(x))
        where f(x) is the feedforward transformation.
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)
        Returns:
            Tensor: Output tensor of shape (batch_size, input_size) after applying the Gated MLP block.
        '''
        gate = self.gate(x)
        x_proj = self.fc2(self.dropout(F.gelu(self.fc1(x))))
        return self.norm(x + gate * x_proj)
    
    def get_x_and_gate(self, x):
        '''
        Get the returned value as well as the gate values from the Gated MLP block
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)
        Returns:
            Tensor: Gate tensor of shape (batch_size, input_size)
        '''
        gate = self.gate(x)
        x_proj = self.fc2(self.dropout(F.gelu(self.fc1(x))))
        return self.norm(x + gate * x_proj), gate

class PredictorGMLP(nn.Module):
    # Predicts the whole transcriptome from the image using a Gated MLP
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.2):
        '''
        Gated MLP based predictor for gene expression from image features.
        Args:
            input_size (int): Size of the input feature vector.
            hidden_size (int): Size of the hidden layers.
            output_size (int): Size of the output gene expression vector.
            num_layers (int, optional): Number of Gated MLP blocks. Defaults to 3.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
        '''
        super().__init__()
        self.layers = nn.ModuleList([
            GatedMLPBlock(input_size, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(input_size, output_size) #make gene expression prediction

    def forward(self, x): 
        '''
        Forward pass for Gated MLP predictor
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)
        Returns:
            Tensor: Output tensor of shape (batch_size, output_size) representing predicted gene expression,
                    or the gated output of shape (num_layers, batch_size, input_size) if return_gate is True.
        '''
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)  # Final gene expression vector
    
    def get_gates(self, x):
        '''
        Get the gate values from each Gated MLP block
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)
        Returns:
            Tensor: Gated output of shape (num_layers, batch_size, input_size)
        '''
        gate_L = []
        for layer in self.layers:
            x, gate = layer.get_x_and_gate(x)
            gate_L.append(gate)
        return torch.stack(gate_L) # Return list of gates from each layer, shape: (num_layers, batch_size, input_size)

#! modules for SPAGHETTI
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        '''
        Backbone of the SPAGHETTI generator network
        Residual block with two convolutional layers and skip connection.
        Args:
            in_channels (int): Number of input and output channels for the residual block.
        '''
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class SpaghettiGenerator(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=9):
        '''
        SPAGHETTI generator network for image-to-image translation.
        The architecture consists of:
            - Initial convolutional layer
            - Downsampling layers (2 layers)
            - Transformation stage with residual blocks
            - Upsampling layers (2 layers)
            - Output convolutional layer
        Args:
            in_channels (int): Number of input and output channels for the images (e.g., 3 for RGB images).
            num_residual_blocks (int, optional): Number of residual blocks in the transformation stage. Defaults to 9.
        '''
        super(SpaghettiGenerator, self).__init__()

        # SPAGHETTI image normalization
        self.normalization = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # Inital Convolution  3*256*256 -> 64*256*256
        out_channels=64
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(in_channels), # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, out_channels, 2*in_channels+1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        channels = out_channels
        
        # Downsampling   64*256*256 -> 128*128*128 -> 256*64*64
        self.down = []
        for _ in range(2):
            out_channels = channels * 2
            self.down += [
                nn.Conv2d(channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.down = nn.Sequential(*self.down)
        
        # Transformation (ResNet)  256*64*64
        self.trans = [ResidualBlock(channels) for _ in range(num_residual_blocks)]
        self.trans = nn.Sequential(*self.trans)
        
        # Upsampling  256*64*64 -> 128*128*128 -> 64*256*256
        self.up = []
        for _ in range(2):
            out_channels = channels // 2
            self.up += [
                nn.Upsample(scale_factor=2), # bilinear interpolation
                nn.Conv2d(channels, out_channels, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.up = nn.Sequential(*self.up)
        
        # Out layer  64*256*256 -> 3*256*256
        self.out = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(channels, in_channels, 2*in_channels+1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.normalization(x)
        x = self.conv(x)
        x = self.down(x)
        x = self.trans(x)
        x = self.up(x)
        x = self.out(x)
        # normalize to range [0,1]
        x = torch.clamp(x, min=-1, max=1)
        min_val = x.min()
        max_val = x.max()
        x = (x-min_val)/(max(max_val-min_val, 1e-5))
        x = torch.clamp(x, min=0, max=1) # ensure no overflow
        return x

#! PCM specific modules for aligning generating pcm predictions
class CellTypeClassifier(nn.Module):
    '''
    Classify the cell types from gene expression features.
    This is used to guide the training of the predictor to better at predicting the gene expression of the pcm
    This is a simple feedforward neural network with two fully connected layers and ReLU activations.
    No softmax activation is applied since it is used in a loss function that applies softmax internally (e.g., CrossEntropyLoss).
    '''
    def __init__(self, input_size, num_classes, hidden_size = 512):
        '''
        Args:
            input_size (int): Size of the input feature vector.
            num_classes (int): Number of cell type classes.
            hidden_size (int, optional): Size of the hidden layers. Defaults to 512.
        '''
        super(CellTypeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)  # Dropout layer to prevent overfitting
        
    def forward(self, x):
        '''
        Forward pass for cell type classifier
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)
        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes) representing class scores for each cell type.
                    Output is in the range of (-inf, inf) as no softmax is applied. Hence use CrossEntropyLoss for training.
        '''
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x