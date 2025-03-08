import torch
import torch.nn as nn

out_size = 576
# out_size = 1600
out_size2 = 512

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


class Classifier(nn.Module):
    def __init__(self, backbone, hdim=512, n_class=10, reg=True):
        super(Classifier, self).__init__()

        # self.backbone = backbone
        self.backbone = nn.Sequential(
            backbone, 
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self.predict = nn.Sequential(
            nn.Linear(backbone.out_features, hdim),
            nn.BatchNorm1d(hdim),
            nn.ReLU(),
            nn.Linear(hdim, n_class)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.predict(x)

        return x
    
    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr},
            {"params": self.predict.parameters(), "lr": 1.0 * base_lr}
        ]

        return params


class ENCODER(nn.Module):
    def __init__(self, rgb=False, resnet=False):
        super(ENCODER, self).__init__()

        if rgb:
            self.encode = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
            )
        else:
            self.encode = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.ReLU(),
                )

        
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


class Classifier(nn.Module):
    def __init__(self, encoder, mlp):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.mlp = mlp
        
    def forward(self, x):
        x = self.encoder(x)
        return self.mlp(x)


class MLP_Encoder(nn.Module):
    def __init__(self, hidden=256):
        super(MLP_Encoder, self).__init__()

        self.encode = nn.Sequential(
        )
        
    def forward(self, x):
        return self.encode(x)
    




import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, backbone, output_channels=3):
        super(Decoder, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(backbone.out_features, 512), 
            nn.ReLU(),
            nn.Linear(512, output_channels * 32 * 32),  # Assuming output image size is 32x32
            nn.ReLU(),
            nn.Unflatten(1, (output_channels, 32, 32)),  
        )
    
    def forward(self, x):
        return self.backbone(x)


class ENCODER_Decoder(nn.Module):
    def __init__(self, rgb=False):
        super(ENCODER_Decoder, self).__init__()
        out_channels = 3 if rgb else 1
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
        # Alternatively, you can add an extra 1x1 conv layer:
        # self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.decode(x)
        # If using final_conv:
        # x = self.final_conv(x)
        return x


class MLP_Decoder(nn.Module):
    def __init__(self, mode, hidden=256):
        super(MLP_Decoder, self).__init__()

        if mode == "mnist":
            dim = 25088
        elif mode == "portraits":
            dim = 32768
        else:
            dim = 2048

        self.decode = nn.Sequential(
            nn.Linear(hidden, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Unflatten(1, (1, 32, 32)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decode(x)



class MLP_Encoder_Decoder(nn.Module):
    def __init__(self, hidden=256):
        super(MLP_Encoder_Decoder, self).__init__()

        self.decode = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 784),  # Assuming output is a flattened 28x28 image
            nn.Unflatten(1, (1, 28, 28)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decode(x)






class VAE(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(VAE, self).__init__()

        self.dim = z_dim

        # encoder
        self.fc1 = nn.Linear(128, z_dim)
        self.fc2 = nn.Linear(128, z_dim)
        self.encode = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(out_size, 128),
            nn.ReLU()
        )
        
        # decoder part
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.fc4 = nn.Linear(648, x_dim)
        
    def encoder(self, x):
        x = self.encode(x)

        return self.fc1(x), self.fc2(x)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        z = z.reshape(-1, self.dim, 1, 1)
        z = self.decode(z)
        z = z.reshape(-1, 648)
        return torch.sigmoid(self.fc4(z)) 
     
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


class CVAE(nn.Module):
    def __init__(self, x_dim, z_dim, res):
        super(CVAE, self).__init__()

        self.dim = z_dim

        # encoder
        self.fc1 = nn.Linear(128, z_dim)
        self.fc2 = nn.Linear(128, z_dim)

        if res:
            self.encode = resnet50(pretrained=True)
            num_features = self.encode.fc.in_features
            self.encode.fc = nn.Linear(num_features, 128)
        else:
            self.encode = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Flatten(),
                nn.Linear(1024, 128),
                nn.ReLU()
            )
        
        # decoder part
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.fc4 = nn.Linear(648, 3*x_dim)
        
    def encoder(self, x):
        x = self.encode(x)

        return self.fc1(x), self.fc2(x)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        z = z.reshape(-1, self.dim, 1, 1)
        z = self.decode(z)
        z = z.reshape(-1, 648)
        # return torch.tanh(self.fc4(z))
        return torch.sigmoid(self.fc4(z)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
