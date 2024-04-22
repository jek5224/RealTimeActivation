import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.idx = None
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        # nn.init.uniform_(self.embeddings.weight, -1 / self.num_embeddings, 1 / self.num_embeddings)   
    def forward(self, inputs):
        # Flatten input
        input_shape = inputs.shape
        flat_inputs = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances between input and embedding vectors
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) + torch.sum(self.embeddings.weight**2, dim=1) - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t()))
        
        # Find closest embedding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        self.idx = encoding_indices
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Quantized
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss # encodings # encodings.permute(0, 2, 1).contiguous()
    
    def get_last_code(self):
        return self.embeddings(self.idx).cpu().detach().numpy()
    
class VAE(nn.Module):
    def __init__(self, 
                 vae_type, # "vq", "beta"
                 num_states, 
                 num_actions, 
                 commitment_cost,
                 condition_dim,
                 encoder_size = [512,512,512],
                 decoder_size = [512,512,512], 
                 num_embeddings = 256, 
                 embedding_dim = 64):
        super(VAE, self).__init__()
        self.vae_type = vae_type 
        self.num_states = num_states
        self.num_actions = num_actions
        self.condition_dim = condition_dim

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = 0.25
        ## Construct Encoder 
        layers = []
        prev_size = num_states
        for size in encoder_size:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_size = size
        if self.vae_type == "vq":
            layers.append(nn.Linear(prev_size, embedding_dim))
        else: # VAE or Beta-VAE
            self.fc_mu = nn.Linear(prev_size, embedding_dim)
            self.fc_var = nn.Linear(prev_size, embedding_dim)
    
        self.encoder = nn.Sequential(*layers)
        if self.vae_type == "vq":
            self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        ## Construct Decoder
        layers = []
        prev_size = embedding_dim + condition_dim
        for size in decoder_size:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_size = size
        layers.append(nn.Linear(prev_size, num_actions))
        self.decoder = nn.Sequential(*layers)

        weights_init(self.encoder)
        weights_init(self.decoder)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        z = self.encoder(x)
        if self.vae_type == "vq":
            quantized, loss = self.quantizer(z)
            return quantized, loss
        else:
            mu = self.fc_mu(z)
            logvar = self.fc_var(z)
            sampled = self.reparameterize(mu, logvar)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return sampled, kld_loss
        
    def forward(self, x):
        z, loss = self.encode(x)
        if len(x.shape) == 2:
            x_recon = self.decoder(torch.concat([x[:, :self.condition_dim], z], dim=-1))
        else:
            x_recon = self.decoder(torch.concat([x[:self.condition_dim], z], dim=-1))
        
        return x_recon, loss * self.beta
