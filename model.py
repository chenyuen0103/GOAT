import torch.nn as nn
import torch

import torch, torch.nn as nn, torch.nn.functional as F

# --- Encoder backbone: any conv trunk that returns N×C×H×W ---
class SmallConv(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28->14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14->7
            nn.Conv2d(64, out_dim, 3, padding=1), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        h = self.pool(self.conv(x)).flatten(1)  # N×out_dim
        return h



class AlgoModel(nn.Module):
    """
    Thin wrapper around a gaussian model to:
      - expose .encoder / .classifier / .mlp for legacy code
      - always return logits from forward()
    """
    def __init__(self, gm: nn.Module):
        super().__init__()
        self.gm = gm
        # Expose expected attrs if code reaches for them
        self.encoder = getattr(gm, "encoder", None)
        # gaussian models may use .classifier or .head — expose both
        self.classifier = getattr(gm, "classifier", getattr(gm, "head", None))
        # some code expects .mlp; if absent, make it a no-op
        self.mlp = getattr(gm, "mlp", nn.Identity())

    def forward(self, x):
        out = self.gm(x)
        # Compat: some gaussian classes optionally return (logits, z)
        return out[0] if isinstance(out, tuple) else out

    def get_parameters(self, base_lr=1.0):
        # If the inner model already defines a param grouping, reuse it.
        if hasattr(self.gm, "get_parameters"):
            return self.gm.get_parameters(base_lr)
        # Fallback: single group
        return [{"params": self.gm.parameters(), "lr": base_lr}]


class Encoder(nn.Module):
    def __init__(self, backbone):
        super(Encoder, self).__init__()

        # self.backbone = backbone
        self.backbone = nn.Sequential(
            backbone, 
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
    
    def forward(self, x):
        x = self.backbone(x)

        return x


# class Classifier(nn.Module):
#     def __init__(self, backbone, hdim=512, n_class=10, reg=True):
#         super(Classifier, self).__init__()

#         # self.backbone = backbone
#         self.backbone = nn.Sequential(
#             backbone, 
#             nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#             nn.Flatten()
#         )
#         self.predict = nn.Sequential(
#             nn.Linear(backbone.out_features, hdim),
#             nn.BatchNorm1d(hdim),
#             nn.ReLU(),
#             nn.Linear(hdim, n_class)
#         )
    
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.predict(x)

#         return x
    
#     def get_parameters(self, base_lr=1.0):
#         """A parameter list which decides optimization hyper-parameters,
#             such as the relative learning rate of each layer
#         """
#         params = [
#             {"params": self.backbone.parameters(), "lr": 0.1 * base_lr},
#             {"params": self.predict.parameters(), "lr": 1.0 * base_lr}
#         ]

#         return params


class ENCODER(nn.Module):
    def __init__(self, rgb=False, resnet=False):
        super(ENCODER, self).__init__()

        if rgb:
            self.encode = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding="same"),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding="same"),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding="same"),
                nn.ReLU(),
            )
            dummy_input = torch.randn(1, 3, 28, 28)
        else:
            self.encode = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding="same"),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding="same"),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding="same"),
                    nn.ReLU(),
                    
                )
            dummy_input = torch.randn(1, 1, 28, 28)

        # Compute output dimension
        with torch.no_grad():
            dummy_output = self.encode(dummy_input)
            self.output_dim = int(torch.flatten(dummy_output, 1).shape[1])

        
    def forward(self, x):
        x = self.encode(x)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, mode, n_class, hidden=1024):
        super(MLP, self).__init__()

        if mode == "mnist":
            dim = 25088
        elif mode == "portraits":
            dim = 32768
        else:
            dim = 2048

        if mode == "covtype":
            hidden = 256
            self.mlp = nn.Sequential(
                nn.Linear(54, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(hidden),
                nn.Linear(hidden, n_class)
            )
        else:
            hidden = 128
            self.mlp = nn.Sequential(
                # nn.BatchNorm2d(32),
                nn.Flatten(),
                # nn.Linear(dim, n_class),
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(hidden),
                nn.Linear(hidden, n_class)
            )

        
    def forward(self, x):
        return self.mlp(x)

class MLPHead(nn.Module):
    def __init__(self, mode, n_class, hidden=1024):
        super(MLPHead, self).__init__()

        if mode == "mnist":
            dim = 25088
        elif mode == "portraits":
            dim = 32768
        else:
            dim = 2048

        if mode == "covtype":
            # hidden = 256
            self.features = nn.Sequential(
                nn.Linear(54, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(hidden)
            )
        else:
            # hidden = 128
            self.features = nn.Sequential(
                # nn.BatchNorm2d(32),
                nn.Flatten(),
                # nn.Linear(dim, n_class),
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(hidden)
            )
        self.classifier = nn.Linear(hidden, n_class)

        
    def forward(self, x):
        z = self.features(x)
        logits = self.classifier(z)
        return logits

class ProjectionMLP(nn.Module):
    """Projection head to produce z for GaussianHead.
       Works for either fmap (N×C×H×W) or vector encoders (N×d)."""
    def __init__(self, encoder, emb_dim=128, in_shape=(1,28,28), hidden=256, pool='gap'):
        super().__init__()
        # probe encoder to infer in_features
        with torch.no_grad():
            dev = next(encoder.parameters()).device
            dummy = torch.zeros(1, *in_shape, device=dev)
            y = encoder(dummy)
        layers = []
        if y.dim() == 4:  # fmap
            C,H,W = y.shape[1:]
            if pool == 'gap':
                layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(1)]
                in_features = C
            else:  # 'flatten'
                layers += [nn.Flatten(1)]
                in_features = C*H*W
        elif y.dim() == 2:  # vector
            in_features = y.shape[1]
        else:
            raise ValueError(f"Unexpected encoder output: {tuple(y.shape)}")
        layers += [nn.Linear(in_features, hidden), nn.ReLU(inplace=True),
                   nn.Linear(hidden, emb_dim), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)  # -> z: N×emb_dim


class Classifier(nn.Module):
    def __init__(self, encoder, mlp):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.mlp = mlp
        
    def forward(self, x):
        x = self.encoder(x)
        return self.mlp(x)
    
class CompressClassifier(nn.Module):
    def __init__(self, classifier, in_dim=25088, out_dim=1024):
        super(CompressClassifier, self).__init__()

        # Keep a handle to the feature extractor (pre-MLP encoder)
        self.encoder = classifier.encoder

        # Linear bottleneck to reduce dimensionality before feeding the head
        self.compressor = nn.Linear(in_dim, out_dim)

        # Build a new head that expects the compressed dimensionality
        head_layers = []
        flatten_layer = nn.Flatten()
        head_layers.append(flatten_layer)

        # Try to reuse the structure of the original MLP (minus the first Linear)
        original_layers = []
        if hasattr(classifier.mlp, 'mlp') and isinstance(classifier.mlp.mlp, nn.Sequential):
            original_layers = list(classifier.mlp.mlp)

        # Remove any leading Flatten from the original sequence
        while original_layers and isinstance(original_layers[0], nn.Flatten):
            original_layers.pop(0)

        if original_layers and isinstance(original_layers[0], nn.Linear):
            first_linear = original_layers.pop(0)
            head_layers.append(nn.Linear(out_dim, first_linear.out_features))
            head_layers.extend(original_layers)
        else:
            # Fallback: simple linear classifier if structure is unexpected
            head_layers.append(nn.Linear(out_dim, classifier.mlp.mlp[-1].out_features))

        self.head = nn.Sequential(*head_layers)
        self.mlp = self.head  # expose for compatibility
        self.emb_dim = out_dim  # expose for compatibility
        self.n_classes = classifier.mlp.mlp[-1].out_features  # expose for compatibility

    def forward(self, x):
        x = self.encoder(x)
        # breakpoint()
        x = x.view(x.size(0), -1)
        x = self.compressor(x)
        return self.head(x)


class MLP_Encoder(nn.Module):
    def __init__(self, hidden=256):
        super(MLP_Encoder, self).__init__()

        self.encode = nn.Sequential(
        )
        
    def forward(self, x):
        return self.encode(x)



class FlattenProj(nn.Module):
    def __init__(self, encoder: ENCODER, emb_dim=128):
        super().__init__()
        self.encoder = encoder
        self.flatten = nn.Flatten(1)
        self.proj = nn.Linear(encoder.output_dim, emb_dim)  # 25088 -> d
        self.act = nn.ReLU()
    def forward(self, x):
        fmap = self.encoder(x)                 # N×32×28×28
        z = self.act(self.proj(self.flatten(fmap)))  # N×d
        return z

# --- Gaussian head: computes Gaussian log-likelihood scores as logits ---
class GaussianHead(nn.Module):
    def __init__(self, n_classes, dim, cov="isotropic"):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(n_classes, dim) * 0.1)
        self.log_pi = nn.Parameter(torch.zeros(n_classes))  # optional learnable priors
        self.cov = cov
        if cov == "isotropic":
            # shared scalar variance per class (or single shared scalar—pick one)
            self.log_var = nn.Parameter(torch.zeros(n_classes))   # per-class σ^2
        elif cov == "diagonal":
            self.log_var = nn.Parameter(torch.zeros(n_classes, dim))  # per-class diag
        else:
            raise ValueError("cov ∈ {'isotropic','diagonal'}")

    def forward(self, z):  # z: N×d
        # returns N×K logits equal to log p(z|y=k) + log π_k up to a constant
        # Gaussian energy: -0.5[(z-μ)^T Σ^{-1} (z-μ) + log det Σ]
        N, d = z.shape
        K = self.mu.shape[0]

        if self.cov == "isotropic":
            # Σ_k = σ_k^2 I
            # (z-μ)^T Σ^{-1} (z-μ) = ||z-μ||^2 / σ_k^2
            # log det Σ_k = d * log σ_k^2
            diff = z[:, None, :] - self.mu[None, :, :]    # N×K×d
            var = self.log_var.exp()                      # K
            quad = (diff.pow(2).sum(dim=2)) / var[None, :]
            logdet = d * self.log_var
            logits = self.log_pi[None, :] - 0.5 * (quad + logdet[None, :])

        else:  # diagonal
            diff = z[:, None, :] - self.mu[None, :, :]    # N×K×d
            var = self.log_var.exp()                       # K×d
            quad = (diff.pow(2) / var[None, :, :]).sum(dim=2)  # N×K
            logdet = self.log_var.sum(dim=1)                    # K
            logits = self.log_pi[None, :] - 0.5 * (quad + logdet[None, :])

        return logits



# class GaussianClassifier(Classifier):
#     """
#     Adds an optional compressor after the encoder:
#       - fmap encoders (N×C×H×W): Conv1x1 to C_comp, optional AvgPool to shrink H,W
#       - vector encoders (N×d):   Linear bottleneck to d_comp
#     Then projects to emb_dim and classifies with a Gaussian head.
#     """
#     def __init__(self, encoder, mlp, gaussian_head, normalize=False):
#         super(GaussianClassifier, self).__init__(encoder, mlp)
#         # check if the encoder 
#         # Gaussian head
#         self.normalize = normalize
#         self.classifier = gaussian_head


#     def forward(self, x):
#         h = self.encoder(x)
#         z = self.mlp.features(h)
#         if self.normalize:
#             z = nn.functional.normalize(z, dim=1, eps=1e-6)
#         # breakpoint()
#         return self.classifier(z)   # logits N×K


class GaussianClassifier(nn.Module):
    def __init__(self, encoder, mlp, gaussian_head,
                 emb_dim: int, n_classes: int, normalize: bool = True,
                 proj_head: nn.Module = None):
        """
        encoder: x -> fmap/vector
        mlp: fmap/vector -> z (N×emb_dim)
        gaussian_head: z -> logits (N×K)
        emb_dim: dimension of z
        n_classes: K
        proj_head: optional SSL projector (z -> q). If None, Identity.
        """
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp           # features only (no classifier)
        self.classifier = gaussian_head   # GaussianHead
        self.normalize = normalize
        self.emb_dim = emb_dim
        self.n_classes = n_classes

        # ---- SSL projector defined here, never inside the loop ----
        self.proj_head = proj_head if proj_head is not None else nn.Identity()

        # ---- EMA class centers defined here, never inside the loop ----
        self.register_buffer("centers", torch.zeros(n_classes, emb_dim))
        self.register_buffer("counts",  torch.zeros(n_classes))

    def features(self, x):
        h = self.encoder(x)
        z = self.mlp(h)
        if self.normalize:
            z = nn.functional.normalize(z, dim=1, eps=1e-6)
        return z

    def forward(self, x):
        z = self.features(x)
        return self.classifier(z)  # logits N×K
