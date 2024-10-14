import torch
import torch.nn as nn
from transformers.modeling_utils import ModuleUtilsMixin
from dataclasses import dataclass

from pdb import set_trace as pds

class LatencyEncoder(nn.Module):
    # Embed scalar to 32 dimensions

    def __init__(self, out_dim = 64):
        super(LatencyEncoder, self).__init__()
        
        # self.device = torch.device("cpu")  # Default device
        self.B = None
        self.fc = nn.Linear(1, out_dim)  # Embed scalar to 32 dimensions
    
    # Fourier feature mapping
    def input_mapping(self, x, B):
        '''
        Args: 
            x (Tensor): batch of latencies, shape [bs, ].
            B (Tensor): embedding vector or scalar, shape [d,] or scalar.
        '''
        if B is None:
            return x
        else:
            # print(x.device)
            # print(B.device)
            # assert False
            x_proj = (2.*torch.pi*x) * B
            # print("np.sin(x_proj): ", np.sin(x_proj))
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)

    def forward(self, x):
        """
        Args: 
            x (ada_sche_hidden): tensor # [bs, seq_len, D]
        """
        x = self.input_mapping(x, self.B)
        x = self.fc(x)
        return x


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class ConcatenateMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latency_dim = config.latency_dim
        self.content_dim = config.content_dim
        self.intermediate_size = 512
        self.hidden_size = 64


        self.up_proj = nn.Linear(self.latency_dim + self.content_dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = nn.ReLU(inplace=True)
        self.act_fn2 = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(p=0.8) # Adjust dropout rate as needed
    
    def out_dim(self):
        return self.hidden_size

    def forward(self, content_features, scalar_embedding):
        """
        Args:
            content_features (float tensor, (bs, seq_len, ada_ScheCfg.content_dim)): feature maps.
            scalar_embedding (float tensor, (bs, ada_ScheCfg.latency_dim)): latency for each input.
        """
        content_features = content_features.mean(dim = 1)  # [bs, ada_ScheCfg.content_dim]
        hidden = torch.cat((content_features, scalar_embedding), dim=1) # [bs, ada_ScheCfg.content_dim + ada_ScheCfg.latency_dim]
        hidden = self.down_proj(self.act_fn(self.up_proj(hidden)))
        embeddings = self.drop_out(self.act_fn2(hidden))
        return embeddings


@dataclass
class ada_SchedulerCfg:
    latency_dim: int = 64
    # latency_Bkey: str = None 
    # latency_gaussian_scale: float = 1.0
    
    content_inp_dim: int = 896 ## feature received from first a hidden states
    content_dim: int = 256
    n_knobs: int = 21 ## hard code for now

    combine_type: str = "concatenate"


    def to_dict(self):
        return {
            "latency_dim": self.latency_dim,
            "content_inp_dim": self.content_inp_dim,
            "content_dim": self.content_dim,
            "n_knobs": self.n_knobs,
            "combine_type": self.combine_type
        }

    

COMBINE_FEATURE_CLASSES = {
    "concatenate": ConcatenateMLP
}
class ada_Scheduler(nn.Module, ModuleUtilsMixin):
    def __init__(self, config: ada_SchedulerCfg):
        super().__init__()
        self.n_knobs = config.n_knobs
        self.latency_encoder = LatencyEncoder(
            config.latency_dim,
            # ada_ScheCfg.latency_Bkey,
            # ada_ScheCfg.latency_gaussian_scale,
        )
        self.content_encoder = nn.Linear(config.content_inp_dim, config.content_dim, bias = False)

        # Combine image and scalar features
        self.combined_fc = COMBINE_FEATURE_CLASSES[config.combine_type](config)
        
        self.out = nn.Linear(self.combined_fc.out_dim(), self.n_knobs)
    
    def construct_embeddings(self, content_features, scalar_embedding):
        """
        Args:
            content_features (float tensor, (bs, seq_len, ada_ScheCfg.content_dim)): feature maps.
            scalar_embedding (float tensor, (bs, ada_ScheCfg.latency_dim)): latency for each input.
        """
        embeddings = self.combined_fc(content_features, scalar_embedding)
        return embeddings

    def forward(self, x, latency):
        """
        Args:
            x (torch.float16, [bs,seq_len, dim]): input features. [4(64), 1024, 896]
            latency (float tensor, (bs,)): latency for each input.
        """
        # print(x.device, x.shape)
        # print(latency.device, latency.shape)
        x = self.content_encoder(x)
        latency = self.latency_encoder(latency.view(-1, 1))
        # print("x.shape: ", x.shape)
        # print("latency.shape: ", latency.shape)
        # assert False

        embeddings = self.construct_embeddings(x, latency)       
        logits = self.out(embeddings)

        return logits