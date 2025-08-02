import torch
import torch.nn as nn
from transformers.models.vits.modeling_vits import VitsEncoder, VitsResidualCouplingBlock
from .vits.models import GeneratorNSF
import math
from transformers import VitsConfig
from .vits.models import SynthesizerTrnMs256NSFsidConfig

class SynthesizerTrnMs_HfVits(nn.Module):
    """
    A hybrid Synthesizer for inference that combines parts of the RVC model
    with modules from the Hugging Face Transformers VITS implementation.

    This model is designed for inference only.
    """
    def __init__(self,
                 config: SynthesizerTrnMs256NSFsidConfig= SynthesizerTrnMs256NSFsidConfig(),
                 vitsConfig: VitsConfig = VitsConfig(hidden_size=192, hidden_dropout=0.0, ffn_dim=768, num_attention_heads=2, ffn_kernel_size=3, layerdrop=0.0, window_size=10, ffn_dropout=0.0, prior_encoder_num_flows=4, prior_encoder_num_wavenet_layers=3, speaker_embedding_size=256),
                 device: str = 'mps' if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else 'cpu',
                 half = True,
                 noise_scale: float = 0.667):
        """
        Initializes the hybrid model.
        """
        super().__init__()
        # Hugging Face modules
        sr = config.sr
        hidden_channels = config.hidden_channels
        self.vistEncoder = VitsEncoder(config=vitsConfig).to(dtype=torch.float16 if half else torch.float32, device=device)
        self.vitsFlow = VitsResidualCouplingBlock(config=vitsConfig).to(dtype=torch.float16 if half else torch.float32, device=device)
        self.hybrid_decoder = GeneratorNSF(
            config.inter_channels,
            config.resblock,
            config.resblock_kernel_sizes,
            config.resblock_dilation_sizes,
            config.upsample_rates,
            config.upsample_initial_channel,
            config.upsample_kernel_sizes,
            gin_channels=config.gin_channels,
            sr=sr)
        self.hybrid_decoder.to(dtype=torch.float16 if half else torch.float32, device=device)

        # RVC modules needed for the hybrid forward pas
        self.gin_channels = config.gin_channels
        # self.hop_length = hop_length#
        self.spk_embed_dim = config.spk_embed_dim
        self.phoneme_embedding = nn.Linear(768, config.hidden_channels).to(dtype=torch.float16 if half else torch.float32, device=device)
        self.emb_pitch = nn.Embedding(256, hidden_channels).to(dtype=torch.float16 if half else torch.float32, device=device)
        self.emb_g = nn.Embedding(self.spk_embed_dim, config.gin_channels).to(dtype=torch.float16 if half else torch.float32) # this is not mpu cast
        
        self.proj = nn.Conv1d(hidden_channels, config.inter_channels * 2, 1).to(dtype=torch.float16 if half else torch.float32,  device=device)

        # RVC layers/constants
        self.hidden_channels = hidden_channels
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.sr = sr
        # Constants for inference
        self.noise_scale = noise_scale
        self.sid = torch.LongTensor([0])  # Placeholder for speaker ID, can be set externally

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor,  nsff0: torch.Tensor):
        """
        Performs the forward pass for inference.
        """
        # --- Speaker Embedding (from RVC) ---
        g = self.emb_g(self.sid).unsqueeze(-1).to(phone.device)

        # --- Encoder (Hybrid RVC Pre-processing + HF Transformer) ---
        x = self.phoneme_embedding(phone) + self.emb_pitch(pitch).to(phone.device)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x_mask_hf = torch.ones(1, x.size(1), 1, device=x.device, dtype=x.dtype)
        x_mask = torch.transpose(x_mask_hf, 1, 2)
        encoder_outputs = self.vistEncoder(
            hidden_states=x,
            padding_mask=x_mask_hf,
            return_dict=False
        )
        x = encoder_outputs[0].transpose(1, 2)
        flow_channels = self.proj.out_channels // 2
        stats = self.proj(x) * x_mask
        m_p, logs_p = torch.split(stats, flow_channels, dim=1)

        # --- Flow and Decoder (HF VITS + RVC NSF) ---
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * self.noise_scale) * x_mask
        z = self.vitsFlow(z_p, x_mask, g, reverse=True)
        o = self.hybrid_decoder(z * x_mask, nsff0, g)[0, 0, :]

        return o