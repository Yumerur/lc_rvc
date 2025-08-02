# (c) Yumerur lisence MIT
import math
import time
from os import PathLike
from typing import ContextManager, Literal, Optional, Union
from transformers import HubertModel
from ..models.configuration_rvc import SynthesizerTrnMs256NSFsidConfig
from transformers import VitsConfig
from ..models.modeling_hybrid import SynthesizerTrnMs_HfVits
import numpy as np
import torch
import noisereduce as nr
import sounddevice as sd
import threading
import queue
from torch.nn import functional as F
from torch import nn
from ..models import RVCFeatureExtractor
import json
from dataclasses import dataclass
from typing import Tuple

@dataclass
class LC_RVC_config:
    """
    Configuration for the LC_RVC model.

    1 size = 160 frames = 10 ms (16k sampling rate)
    """
    chunk_size: int = 40 # 0.4 seconds
    total_context_size: int = 200 # 2 seconds
    raw_audio_padding_length: int = 10 # 0.1 seconds
    left_size: int = 2 # 0.02 seconds, to avoid edge effects intentionally left clips for next chunk

class LC_RVC(nn.Module):
    def __init__(self, 
                 hubert: HubertModel, 
                 vits: SynthesizerTrnMs_HfVits, 
                 config: LC_RVC_config):
        
        super().__init__()
        self.config = config
        
        # Extract all hubert modules
        self.hubert_feature_extractor = hubert.feature_extractor
        self.hubert_encoder = hubert.encoder
        self.hubert_feature_projection = hubert.feature_projection

        self.hubert = hubert # to use _mask hidden states

        # Extract VITS/RVC modules
        self.vits_encoder = vits.vistEncoder
        self.vits_flow = vits.vitsFlow
        self.hybrid_decoder = vits.hybrid_decoder
        
        # RVC embedding modules (need to extract from the hybrid model)
        self.phoneme_embedding = vits.phoneme_embedding
        self.emb_pitch = vits.emb_pitch
        self.emb_g = vits.emb_g
        self.proj = vits.proj
        self.lrelu = vits.lrelu
        self.noise_scale = vits.noise_scale
        self.sid = vits.sid
        self.hidden_channels = vits.hidden_channels

        # Context buffers for large context attention
        self.context_length = config.total_context_size

        self.process_size = ((config.chunk_size + config.raw_audio_padding_length * 2) * 160 - 400) // 320 + 1  # Convert chunk size to samples
        self.process_size_f0 = self.process_size * 2

        self.hubert_context_length = config.total_context_size // 2
        self.hubert_hidden_context = torch.zeros(
            1, self.hubert_context_length, hubert.config.hidden_size, device=hubert.device, dtype=hubert.dtype
        )
        self.f0_context = torch.zeros(
            1, self.context_length, vits.hidden_channels, device=hubert.device, dtype=hubert.dtype
        )
        
        self.valid_context = 0

    @torch.no_grad()
    def register_context(self, new_audio: torch.Tensor, pitch: torch.Tensor):
        """
        Register new audio and pitch context for the next forward pass.
        This is used to pre-populate context when resuming from silence.

        (for use in main loop remain right padding)
        
        Args:
            new_audio (torch.Tensor): Audio tensor to register as context [1, audio_length] - should NOT be padded
            pitch (torch.Tensor): Pitch tensor extracted from F0 feature extraction [1, pitch_length] - should be padded
        """
        # Extract HuBERT features from unpadded new audio (input should already be unpadded)
        # new_audio_size ~ raw_pad * 160 + chunk_size * 160 + raw_pad * 160
        new_audio_features = self.hubert_feature_extractor(new_audio).transpose(1, 2)
        new_audio_features = self.hubert_feature_projection(new_audio_features)
    

        # HuBERT downsamples by 320/160 = 2x, so audio frames become feature frames
        # FIX: USE same padding size for raw_padding length
        hubert_expression_pad_frames = self.config.raw_audio_padding_length // 2
        
        valid_context_for_hubert = self.valid_context // 2
        # if valid context is 0, with padding frames
        if valid_context_for_hubert == 0:
            # do not register right padding but remain right
            valid_context_for_hubert = self.process_size
            # right padding is hold in the context
            self.hubert_hidden_context[:, -valid_context_for_hubert:, :] = new_audio_features
        else:
            # register with shifting context (pay attention to maintain total size)
            valid_context_for_hubert = valid_context_for_hubert + (self.process_size - hubert_expression_pad_frames * 2)
            self.hubert_hidden_context = self.hubert_hidden_context.roll(shifts=-(self.process_size - hubert_expression_pad_frames * 2), dims=1)
            # leave the right padding
            self.hubert_hidden_context[:, -(self.process_size - hubert_expression_pad_frames):, :] = new_audio_features[:, hubert_expression_pad_frames:, :]

        # For F0 context, remove padding from pitch tensor to match the unpadded audio chunk
        new_f0_embedding = self.emb_pitch(pitch).to(new_audio.device, dtype=new_audio.dtype)
        f0_expression_pad_frames = self.config.raw_audio_padding_length

        # same process as HuBert
        if self.valid_context == 0:
            # do not register right padding but remain right
            self.valid_context = self.process_size_f0
            self.f0_context[:, -self.valid_context:, :] = new_f0_embedding
        else:
            # register with shifting context (pay attention to maintain total size)
            self.valid_context = self.valid_context + (self.process_size_f0 - f0_expression_pad_frames * 2)
            self.f0_context = self.f0_context.roll(shifts=-(self.process_size_f0 - f0_expression_pad_frames * 2), dims=1)
            # leave the right padding
            self.f0_context[:, -(self.process_size_f0 - f0_expression_pad_frames):, :] = new_f0_embedding[:, f0_expression_pad_frames:, :]
    
    def forward(self, new_audio: torch.Tensor, pitch: torch.Tensor, nsff0: torch.Tensor):
        """
        Forward pass for the LC_RVC model with large context attention.
        
        This implementation provides longer context for both HuBERT and RVC encoders
        while maintaining low latency by only processing new chunks.
        
        Args:
            new_audio (torch.Tensor): New audio input tensor [1, chunk_size * 160]
            pitch (torch.Tensor): Pitch tensor [1, chunk_size]
            nsff0 (torch.Tensor): NSFF0 tensor for decoder [1, chunk_size * 160]
        
        Returns:
            torch.Tensor: Output of the hybrid decoder
        """
        
        # =====================================================
        # Step 1: Extract HuBERT features from new audio chunk
        # =====================================================        
        # Extract HuBERT features from padded new audio
        self.register_context(new_audio, pitch) # type: ignore
        
        # =====================================================
        # Step 3: Pass through HuBERT encoder with full context
        # =====================================================
        
        hubert_context = self.hubert_hidden_context  # Use clone to avoid modifying the original context
        
        if self.valid_context // 2 < self.hubert_context_length:
            attention_mask = torch.zeros(1, hubert_context.size(1), 1, device=hubert_context.device, dtype=torch.bool)
            attention_mask[:, -self.valid_context // 2:, :] = True  # Mask right padding
        else:
            attention_mask = torch.ones(1, hubert_context.size(1), 1, device=hubert_context.device, dtype=torch.bool)

        attention_mask_hf = self.hubert._get_feature_vector_attention_mask(self.hubert_context_length, attention_mask)

        hubert_outputs = self.hubert_encoder(
            hidden_states=hubert_context,
            attention_mask=attention_mask_hf,
            return_dict=False
        )
        full_hubert_hidden = hubert_outputs[0]
        
        # =====================================================
        # Step 4: Upsample HuBERT output to match F0 sequence length
        # =====================================================
        
        # F0 embeddings have 2x the sequence length of HuBERT output
        # Use repeat_interleave to match F0 sequence length
        full_hubert_hidden = full_hubert_hidden.repeat_interleave(2, dim=1)

        # Update padding mask to match the upsampled sequence length
        vits_padding_mask = attention_mask.repeat(1, 2, 1)
        vits_attention_mask  = vits_padding_mask.squeeze(-1)  # [1, seq_len, 1] for VITS encoder
        
        # =====================================================
        # Step 5: Extract phoneme embeddings and prepare F0
        # =====================================================
        
        # Convert HuBERT outputs to phoneme embeddings
        full_phoneme_embedding = self.phoneme_embedding(full_hubert_hidden)
        
        # Add phoneme and F0 embeddings
        combined_hidden = full_phoneme_embedding + self.f0_context
        combined_hidden = combined_hidden * math.sqrt(self.hidden_channels)
        combined_hidden = self.lrelu(combined_hidden)
        
        # Pass through VITS encoder
        # Note: The transformers VitsEncoder expects [batch, sequence, features] input format
        vits_encoder_outputs = self.vits_encoder(
            hidden_states=combined_hidden,  # [1, seq_len, hidden_channels]
            padding_mask=vits_padding_mask,
            attention_mask=vits_attention_mask,
            return_dict=False
        )
        encoder_output = vits_encoder_outputs[0].transpose(1, 2)  # [1, hidden_dim, seq_len]
        
        # =====================================================
        # Step 9: Extract only the new chunk for decoding
        # =====================================================
        
        end_idx_decode = -self.config.left_size
        start_idx_decode = - self.config.left_size - self.process_size_f0

        x = encoder_output[:, :, start_idx_decode:end_idx_decode]
        x_mask = torch.ones(1, 1, x.size(2), device=encoder_output.device, dtype=encoder_output.dtype)
        
        # =====================================================
        # Step 10: Flow and Decoder processing
        # =====================================================
        
        # Project to flow channels
        flow_channels = self.proj.out_channels // 2
        stats = self.proj(x)
        m_p, logs_p = torch.split(stats, flow_channels, dim=1)
        
        # Sample from prior distribution
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * self.noise_scale)
        
        # Speaker embedding
        g = self.emb_g(self.sid).unsqueeze(-1).to(new_audio.device)
        
        # Flow processing
        z = self.vits_flow(z_p, x_mask, g, reverse=True)
        
        # Decoder processing
        output = self.hybrid_decoder(z * x_mask, nsff0, g)[0, 0, :]
        
        # unpadding the output to match the original chunk size
        
        # =====================================================
        # Step 11: Update context buffers -> its done in register_context
        # =====================================================
        return output

    def reset_context(self):
        """
        Reset the hidden state and f0 context.
        """
        self.hubert_hidden_context = torch.zeros_like(self.hubert_hidden_context)
        self.f0_context = torch.zeros_like(self.f0_context)
        self.valid_context = 0

class AGC:
    """Ultra-fast time-domain AGC for real-time voice conversion with anti-clipping protection."""
    
    def __init__(self, target_level=0.3, smoothing=0.95, soft_limit_threshold_rate: float = 1.8, soft_limit_ratio=0.3):
        self.target_level = target_level
        self.smoothing = smoothing
        self.previous_gain = 1.0
        # Soft limiter parameters to prevent cracking
        self.soft_limit_threshold = target_level * soft_limit_threshold_rate  # Start gentle compression at 80% of max
        self.soft_limit_ratio = soft_limit_ratio  # Gentle compression ratio (0.3 = mild compression)
    
    def apply_soft_limiter(self, audio_chunk):
        """Apply gentle curve to prevent sound cracking."""
        # Calculate peak level of the chunk
        peak_level = np.max(np.abs(audio_chunk))
        
        if peak_level <= self.target_level * 1.05:
            # Below threshold - no limiting needed
            return audio_chunk
        
        # Calculate gentle compression curve
        # Use a smooth tanh-based curve for natural sound
        overshoot = peak_level - self.target_level
        max_overshoot = self.soft_limit_threshold - self.target_level
        
        if max_overshoot > 0:
            # Normalize overshoot (0 to 1)
            normalized_overshoot = overshoot / max_overshoot
            
            # Apply gentle compression using tanh curve
            # This provides smooth, natural-sounding compression
            compressed_overshoot = np.tanh(normalized_overshoot * 2.0) * self.soft_limit_ratio
            
            # Calculate the reduction factor
            target_peak = self.soft_limit_threshold + (compressed_overshoot * max_overshoot)
            reduction_factor = target_peak / peak_level if peak_level > 0 else 1.0
            
            # Apply gentle reduction
            return audio_chunk * reduction_factor
        
        return audio_chunk
    
    def process(self, audio_chunk):
        """Process audio chunk with AGC and anti-clipping protection."""
        if len(audio_chunk) == 0:
            return audio_chunk
        
        # Calculate chunk RMS
        current_rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Calculate required gain
        if current_rms > 1e-8:
            target_gain = self.target_level / current_rms
            target_gain = np.clip(target_gain, 0.2, 3.0)  # Limit gain range
        else:
            target_gain = 1.0
        
        # Smooth gain transition
        smoothed_gain = (self.smoothing * self.previous_gain + 
                        (1 - self.smoothing) * target_gain)
        self.previous_gain = smoothed_gain
        
        # Apply AGC gain
        agc_output = audio_chunk * smoothed_gain
        
        # Apply gentle curve to prevent cracking
        protected_output = self.apply_soft_limiter(agc_output)
        
        return protected_output

def openModel(file):
    from pathlib import Path
    modelDir = Path("ModelFolder").joinpath(file)
    weight = torch.load(modelDir.joinpath("model.pth"), map_location="cpu", weights_only=True)
    with open(modelDir.joinpath("metadata.json"), "r") as f:
        config = json.load(f)
    modelConfig = SynthesizerTrnMs256NSFsidConfig(
        *config
    )
    output_sampling_rate = config[-1]  
    vitsConfig = VitsConfig(hidden_size=192, hidden_dropout=0.0, ffn_dim=768, num_attention_heads=2, ffn_kernel_size=3, layerdrop=0.0, window_size=10, ffn_dropout=0.0, prior_encoder_num_flows=4, prior_encoder_num_wavenet_layers=3, speaker_embedding_size=256)
    return modelConfig, vitsConfig, weight, output_sampling_rate

class PerfCounter(ContextManager):
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __enter__(self) -> "PerfCounter":
        self._start_time = time.time()
        return super().__enter__()

    def __exit__(self, *args) -> None:
        self.elapsed = time.time() - self._start_time
        if self.name:
            print(f"{self.name}:\t{self.elapsed:0.3f}s")


def get_volumes(audio_input: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate RMS-based volume levels for consistent scaling with output normalization.
    
    This function now uses RMS (Root Mean Square) calculation instead of linear peak
    levels to match the RMS-based output normalization, preventing volume scaling
    mismatches that can cause popping sounds.
    
    Args:
        audio_input (np.ndarray): Input audio samples
        window_size (int): Window size for volume calculation
        
    Returns:
        np.ndarray: RMS volume levels for each window
    """
    output_length = len(audio_input) // window_size
    x = audio_input[: output_length * window_size]
    x = np.reshape(x, (output_length, window_size))
    # Calculate RMS (Root Mean Square) instead of linear peak
    x = np.sqrt(np.mean(x**2, axis=1))
    return x


class AudioStreamer:
    """Enhanced audio streaming using sounddevice with exact chunk size buffering for RVC compatibility."""
    
    def __init__(self, input_sr: int, output_sr: int, input_device: Optional[int] = None, 
                 output_device: Optional[int] = None, chunk_size: int = 6400):
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.input_device = input_device
        self.output_device = output_device
        self.target_chunk_size = chunk_size  # Target chunk size for RVC processing
        
        # Calculate sounddevice blocksize (smaller for low latency)
        self.blocksize = min(2048, chunk_size // 4)  # Use smaller blocks for better responsiveness
        
        # Audio queues for exact chunk sizes
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Input buffering for exact chunk assembly
        self.input_buffer = np.array([], dtype=np.float32)
        self.output_buffer = np.array([], dtype=np.float32)
        
        # Control
        self.running = threading.Event()
        self.input_stream = None
        self.output_stream = None
        
    def input_callback(self, indata: np.ndarray, frames: int, time, status):
        """Callback for input audio stream with exact chunk size buffering."""
        if status:
            print(f"Input stream status: {status}")
        
        # Add new audio data to buffer
        new_audio = indata[:, 0].copy()  # Take first channel
        self.input_buffer = np.concatenate([self.input_buffer, new_audio])
        
        # Extract complete chunks of target size
        while len(self.input_buffer) >= self.target_chunk_size:
            # Extract exact chunk
            chunk = self.input_buffer[:self.target_chunk_size].copy()
            self.input_buffer = self.input_buffer[self.target_chunk_size:]
            
            try:
                # Put exact-sized chunk in queue (non-blocking)
                self.input_queue.put_nowait(chunk)
            except queue.Full:
                # Drop frame if queue is full to prevent latency buildup
                pass
    
    def output_callback(self, outdata: np.ndarray, frames: int, time, status):
        """Callback for output audio stream with buffering for smooth playback."""
        if status:
            print(f"Output stream status: {status}")
        
        # Try to get new processed audio and add to buffer
        try:
            while not self.output_queue.empty():
                new_output = self.output_queue.get_nowait()
                self.output_buffer = np.concatenate([self.output_buffer, new_output])
        except queue.Empty:
            pass
        
        # Output audio from buffer
        if len(self.output_buffer) >= frames:
            # Use buffered audio
            outdata[:, 0] = self.output_buffer[:frames]
            self.output_buffer = self.output_buffer[frames:]
        else:
            # Not enough buffered audio - output what we have + silence
            if len(self.output_buffer) > 0:
                outdata[:len(self.output_buffer), 0] = self.output_buffer
                outdata[len(self.output_buffer):, 0] = 0
                self.output_buffer = np.array([], dtype=np.float32)
            else:
                # Output silence if no data available
                outdata.fill(0)
    
    def start(self):
        """Start audio streaming."""
        self.running.set()
        
        # Reset buffers
        self.input_buffer = np.array([], dtype=np.float32)
        self.output_buffer = np.array([], dtype=np.float32)
        
        # Start input stream
        self.input_stream = sd.InputStream(
            device=self.input_device,
            channels=1,
            samplerate=self.input_sr,
            blocksize=self.blocksize,
            callback=self.input_callback,
            dtype=np.float32
        )
        
        # Calculate output blocksize to match timing
        output_blocksize = int(self.blocksize * self.output_sr / self.input_sr)
        
        # Start output stream
        self.output_stream = sd.OutputStream(
            device=self.output_device,
            channels=1,
            samplerate=self.output_sr,
            blocksize=output_blocksize,
            callback=self.output_callback,
            dtype=np.float32
        )
        
        self.input_stream.start()
        self.output_stream.start()
        
        print(f"Audio streaming started:")
        print(f"  Input: {self.input_sr}Hz, blocksize={self.blocksize}, target_chunk_size={self.target_chunk_size}")
        print(f"  Output: {self.output_sr}Hz, blocksize={output_blocksize}")
    
    def stop(self):
        """Stop audio streaming."""
        self.running.clear()
        
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        
        print("Audio streaming stopped.")
    
    def get_input_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get input audio data with exact target chunk size."""
        try:
            chunk = self.input_queue.get(timeout=timeout)
            # Ensure exact size (should always be true with proper buffering)
            assert len(chunk) == self.target_chunk_size, f"Expected {self.target_chunk_size}, got {len(chunk)}"
            return chunk
        except queue.Empty:
            return None
    
    def put_output_audio(self, audio_data: np.ndarray, timeout: float = 0.1) -> bool:
        """Put processed audio for output."""
        try:
            self.output_queue.put(audio_data, timeout=timeout)
            return True
        except queue.Full:
            return False


def normalize_chunk_rms(audio_chunk: np.ndarray, target_rms: float, current_rms: Optional[float] = None) -> np.ndarray:
    """
    Normalize audio chunk RMS to match target RMS level.
    
    This mimics the global RMS normalization mechanism from the original realtime_vcEZ.py
    to prevent amplification of breathing/noise during silence hold time.
    
    Args:
        audio_chunk (np.ndarray): Input audio chunk
        target_rms (float): Target RMS level to normalize to
        current_rms (Optional[float]): Current RMS if already calculated, otherwise will be computed
    
    Returns:
        np.ndarray: Normalized audio chunk
    """
    # Calculate current RMS if not provided
    rms_value = current_rms if current_rms is not None else np.sqrt(np.mean(audio_chunk**2))
    
    # Avoid division by zero
    if rms_value > 1e-8:
        # Calculate normalization factor
        normalization_factor = target_rms / rms_value
        # Limit normalization to prevent excessive amplification (max 3x)
        normalization_factor = min(normalization_factor, 3.0)
        return audio_chunk * normalization_factor
    else:
        return audio_chunk


def unroll_mean(x, y, window):
    return np.repeat(x, window).reshape(-1, window).mean(axis=1)[: y.shape[0] * window]

from pathlib import Path
p = Path(__file__).parent.parent.parent / "hubert_base_hf"

# Function to check if a given RVC size is achievable with the constraints
def realtime_vc(
    modelname: str = "",
    expectChunk: float = 0.4,
    contextLentgh: float = 2.0,  # Total context length in seconds
    f0_method: Literal["pm", "harvest"] = "pm", 
    f0_up_key: float = 0,
    input_device_index: Optional[int] = None,
    output_device_index: Optional[int] = None,
    volume: float = 1.0,
    expected_audio_level: float = 0.03,
    device: Union[str, torch.device] = "mps",
    fp16: bool = True,
    noise_detection_threshold: float = 0.03,  # Threshold for noise detection
    silence_hold_time: float = 0.5,  # Time to continue processing after volume drops below threshold (seconds)
    audio_hold_time: float = 0.1,  # Time to wait before resuming processing after silence ends (seconds)
    console_debug: bool = False
) -> None:
    """
    Real-time voice conversion function with Large Context RVC (LC_RVC).
    
    This implementation provides superior long-tone handling and natural voice conversion
    by maintaining large attention context for both HuBERT and RVC encoders while keeping
    low latency through chunked processing.
    
    Key Features:
    - Large context attention (2 seconds) for better long-tone understanding
    - Low latency chunked processing (0.4 seconds chunks)
    - Efficient context management with sliding window
    - Proper attention masking for padding regions
    - Real-time audio streaming
    
    Args:
        modelname (str): Name of the RVC model to use
        expectChunk (float): Expected chunk size in seconds (default: 0.4s)
        f0_method (Literal["pm", "harvest"]): F0 extraction method
        f0_up_key (float): Key to transpose F0
        input_device_index (Optional[int]): Input audio device index
        output_device_index (Optional[int]): Output audio device index
        volume (float): Volume scaling factor
        expected_audio_level (float): Expected audio level for normalization
        device (Union[str, torch.device]): Device to run the model on
        fp16 (bool): Use FP16 precision if available
        silence_hold_time (float): Time to continue processing after silence is detected
        audio_hold_time (float): Time to wait before resuming processing after silence ends
    
    Returns:
        None: This function runs indefinitely until stopped manually.
    """
    print("Loading HuBERT model...")
    hubert = HubertModel.from_pretrained(p.absolute())  # type: ignore
    #hubert: HubertModel = HubertModel.from_pretrained(p.absolute()) # type: ignore
    # Move model to device and set dtype
    hubert = hubert.to(device)  # type: ignore
    if fp16:
        hubert = hubert.half()
    
    print(f"Loading RVC model: {modelname}")
    modelConfig, vitsConfig, weight, output_sampling_rate = openModel(modelname)
    
    print("Creating SynthesizerTrnMs_HfVits...")
    device_str = str(device) if not isinstance(device, str) else device
    vits = SynthesizerTrnMs_HfVits(
        config=modelConfig,
        vitsConfig=vitsConfig,
        device=device_str,
        half=fp16,
        noise_scale=0.667
    )
    vits.load_state_dict(weight, strict=False)
    vits.eval()
    
    print("Creating LC_RVC configuration...")
    # Configure LC_RVC for optimal performance
    lc_config = LC_RVC_config(
        chunk_size=int(expectChunk * 100),  # Convert to 10ms frames
        total_context_size=int(contextLentgh * 100),  # 2 seconds of context
        raw_audio_padding_length=10,  # 100ms padding for edge effects
        left_size=2  # 20ms left clip to avoid edge effects
    )
    
    print("Creating LC_RVC model...")
    lc_rvc = LC_RVC(hubert, vits, lc_config)
    lc_rvc.eval()

    torch.compile(lc_rvc, mode="reduce-overhead", fullgraph=True)
    
    test_input = torch.randn(1, int((lc_config.chunk_size + lc_config.raw_audio_padding_length * 2) * 160), dtype=torch.float16).to(device)
    audio_len = (int((lc_config.chunk_size + lc_config.raw_audio_padding_length * 2) * 160 - 400) // 320 + 1) * 2
    test_pitch = torch.randint(2,255, (1, audio_len)).to(device)
    test_nsff0 = torch.randn(1, audio_len, dtype=torch.float16).to(device)

    print(test_input.shape, test_pitch.shape, test_nsff0.shape)
    with torch.no_grad():
        # Warm up the model with a test input
        lc_rvc(test_input, test_pitch, test_nsff0)
    
    # Initialize feature extractor for F0 processing
    feature_extractor = RVCFeatureExtractor(f0_method=f0_method)
    
    # Audio processing configuration
    input_sr = 16000  # HuBERT sampling rate
    chunk_frames = int(expectChunk * input_sr)
    
    print("Initializing audio streaming...")
    audio_streamer = AudioStreamer(
        input_sr=input_sr,
        output_sr=output_sampling_rate,
        input_device=input_device_index,
        output_device=output_device_index,
        chunk_size=chunk_frames
    )
    
    # Initialize audio processing components
    agc = AGC(
            target_level=volume * expected_audio_level, 
            smoothing=0.95,
            soft_limit_threshold_rate=1.5,  # Start gentle compression at 80% of max
            soft_limit_ratio=0.3       # Gentle compression to prevent cracking
        )
    
    # Audio buffers 
    previous_chunk = None  # Store previous chunk for context after silence
    silence_counter = 0
    processing_enabled = True
    
    # Global RMS tracking to prevent amplification of breathing/noise during hold time
    global_rms = None  # RMS calculated when volume is above threshold
    audio_chunks_count = 0  # Count of chunks with audio above threshold
    silence_chunks_count = 0  # Count of chunks with audio below threshold
    
    print("Starting real-time voice conversion with Large Context RVC...")
    print("Press Ctrl+C to stop.")
    
    try:
        audio_streamer.start()
        
        with torch.no_grad():
            while True:
                # Get input audio
                input_chunk = audio_streamer.get_input_audio()
                if input_chunk is None:
                    continue
                
                # normalize input chunk to expected audio level
                input_chunk = input_chunk - np.mean(input_chunk)  # Remove DC offset

                # Store previous chunk for context recovery after silence
                if processing_enabled:
                    previous_chunk = input_chunk

                # Check for silence and manage global RMS using combined_audio (before padding)
                volume_level = np.sqrt(np.mean(input_chunk**2))

                # Determine if volume is above threshold
                volume_above_threshold = volume_level >= expected_audio_level * noise_detection_threshold
                
                # Global RMS management to prevent amplification of breathing/noise during hold time
                if volume_above_threshold:
                    # Audio is above threshold - update global_rms with current chunk
                    if global_rms is None:
                        # First time above threshold - initialize global_rms
                        global_rms = volume_level
                        if console_debug:
                            print(f"Initializing global_rms: {global_rms:.6f}")
                    else:
                        # Continue integration of global_rms using exponential moving average
                        # This provides smooth transitions and prevents sudden RMS changes
                        global_rms_alpha = 0.1  # Smoothing factor (0.1 = slow adaptation)
                        global_rms = (1.0 - global_rms_alpha) * global_rms + global_rms_alpha * volume_level
                    
                    audio_chunks_count += 1
                    silence_chunks_count = 0  # Reset silence counter when audio detected
                    
                elif processing_enabled and global_rms is not None:
                    # In hold time (volume below threshold but still processing)
                    # Normalize using existing global_rms to prevent amplification of breathing/noise
                    silence_chunks_count += 1
                    # Apply normalization to the combined_audio (not padded_audio)
                    input_chunk = input_chunk / global_rms if global_rms > 1e-8 else input_chunk

                    padding_length = int(lc_config.raw_audio_padding_length * 0.01 * input_sr)

                    # Re-create padded audio with normalized data
                    padded_audio = np.pad(input_chunk, (padding_length, padding_length), mode='reflect')
                    # print(f"Normalizing during hold time (chunk {silence_chunks_count}): target_rms={global_rms:.6f}")
                else:
                    silence_chunks_count += 1
                
                if volume_level < expected_audio_level * 0.1:  # Silence threshold
                    silence_counter += 1
                    if silence_counter > int(silence_hold_time / expectChunk):
                        if processing_enabled:  # Only log when switching from enabled to disabled
                            processing_enabled = False
                            # Reset global_rms when exiting hold time to silence mode
                            global_rms = None
                            audio_chunks_count = 0
                            silence_chunks_count = 0
                            # Keep previous_chunk for context recovery (don't reset it)
                            if console_debug:
                                print("Switching to silence mode, global_rms and chunk counters reset")
                        # Output silence
                        silence_output = np.zeros(int(expectChunk * output_sampling_rate))
                        audio_streamer.put_output_audio(silence_output)
                        continue
                else:
                    if not processing_enabled:
                        # Wait a bit before resuming to avoid noise
                        if silence_counter < int(audio_hold_time / expectChunk):
                            silence_counter += 1
                            silence_output = np.zeros(int(expectChunk * output_sampling_rate))
                            audio_streamer.put_output_audio(silence_output)
                            continue
                        else:
                            processing_enabled = True
                            lc_rvc.reset_context()  # Reset context after silence
                            global_rms = None  # Reset global_rms when resuming from silence
                            audio_chunks_count = 0
                            silence_chunks_count = 0
                            
                            # Use previous chunk for context if available
                            if previous_chunk is not None:
                                # Normalize previous chunk using current chunk's RMS for context
                                current_chunk_rms = np.sqrt(np.mean(input_chunk**2))
                                if current_chunk_rms > 1e-8:
                                    # Normalize previous chunk to match current chunk's level
                                    normalized_previous_chunk = normalize_chunk_rms(previous_chunk, current_chunk_rms)
                                    
                                    # Extract F0 features for the normalized previous chunk (with padding for proper F0 extraction)
                                    try:
                                        excepted_f0_length = (int((lc_config.chunk_size + lc_config.raw_audio_padding_length * 2) * 160 - 400) // 320 + 1) * 2 # F0 hop length is typically 160 samples

                                        # Add padding for F0 extraction (same as main processing)
                                        prev_padding_length = int(lc_config.raw_audio_padding_length * 160)
                                        padded_previous_chunk = np.pad(normalized_previous_chunk, (prev_padding_length, prev_padding_length), mode='reflect')
                                        
                                        # Extract F0 features from padded previous chunk
                                        f0_coarse_prev, f0_prev = feature_extractor._extract_f0_features(
                                            padded_previous_chunk, f0_up_key=f0_up_key
                                        )
                                                
                                        # Convert to tensors and ensure proper dtype/device consistency
                                        prev_audio_tensor = torch.from_numpy(padded_previous_chunk).float().unsqueeze(0).to(device)
                                        prev_pitch_tensor = torch.from_numpy(f0_coarse_prev[:, :excepted_f0_length]).int().to(device)
                                        
                                        #print(excepted_f0_length, prev_pitch_tensor.shape )
                                        # Ensure proper dtype for half precision if needed
                                        if fp16:
                                            prev_audio_tensor = prev_audio_tensor.half()
                                        
                                        # Register context using the normalized previous chunk (pass unpadded audio to register_context)
                                        lc_rvc.register_context(prev_audio_tensor, prev_pitch_tensor)
                                        
                                        # Update left audio buffer with previous chunk
                                        #print(f"Resuming with previous chunk context, normalized to RMS: {current_chunk_rms:.6f}")
                                        
                                    except Exception as e:
                                        print(f"Failed to extract F0 for previous chunk: {e}")
                                        print("Resuming processing without previous context")
                                else:
                                    print("Resuming processing, context and global_rms reset, chunk counters reset")
                            else:
                                print("Resuming processing, context and global_rms reset, chunk counters reset")
                    silence_counter = 0
                
                if not processing_enabled:
                    continue
                
                # Normalize input chunk to expected audio le
                # Apply normalization to the combined_audio (not padded_audio)
                input_chunk = input_chunk / volume_level if volume_level > 1e-8 else input_chunk

                padding_length = int(lc_config.raw_audio_padding_length * 0.01 * input_sr)

                # Re-create padded audio with normalized data
                padded_audio = np.pad(input_chunk, (padding_length, padding_length), mode='reflect')

                # Prepare tensors for LC_RVC using the padded audio (which includes both combined_audio + padding)
                audio_tensor = torch.from_numpy(padded_audio).float().unsqueeze(0).to(device)
                if fp16:
                    audio_tensor = audio_tensor.half()
                
                    # Extract F0 features using the padded audio for better edge handling
                try:
                    f0_coarse, f0 = feature_extractor._extract_f0_features(
                        padded_audio, f0_up_key=f0_up_key
                    )
                    pitch_tensor = torch.from_numpy(f0_coarse).int().to(device)
                    nsff0_tensor = torch.from_numpy(f0).float().to(device)
                    if fp16:
                        nsff0_tensor = nsff0_tensor.half()
                    
                    # Calculate expected sequence length for padded audio (0.4s + 2*0.1s padding = 0.6s total)
                    # Expected F0 frames = (padded_audio_samples // hop_length)
                    expected_f0_length = (int((lc_config.chunk_size + lc_config.raw_audio_padding_length * 2) * 160 - 400) // 320 + 1) * 2 # F0 hop length is typically 160 samples
                    
                    # Ensure F0 tensors match expected length
                    if pitch_tensor.size(1) > expected_f0_length:
                        pitch_tensor = pitch_tensor[:, :expected_f0_length]
                        nsff0_tensor = nsff0_tensor[:, :expected_f0_length]
                    elif pitch_tensor.size(1) < expected_f0_length:
                        # Pad if too short
                        pad_length = expected_f0_length - pitch_tensor.size(1)
                        pitch_tensor = torch.cat([pitch_tensor, pitch_tensor[:, -1:].expand(-1, pad_length)], dim=1)
                        nsff0_tensor = torch.cat([nsff0_tensor, nsff0_tensor[:, -1:].expand(-1, pad_length)], dim=1)
                    
                except Exception as e:
                    print(f"F0 extraction error: {e}")
                    continue
                
                # Process with LC_RVC
                try:
                    start_time = time.time()
                    output = lc_rvc(audio_tensor, pitch_tensor, nsff0_tensor)
                    process_time = time.time() - start_time
                    
                    # Convert to numpy and apply volume
                    output_audio = output.cpu().float().numpy() * volume
                    
                    # unpad output audio to match expected chunk size
                    output_audio = output_audio[lc_config.raw_audio_padding_length * (output_sampling_rate // 100): -lc_config.raw_audio_padding_length * (output_sampling_rate // 100)]

                    # Ensure correct length
                    expected_output_len = int(expectChunk * output_sampling_rate)
                    if len(output_audio) != expected_output_len:
                        if len(output_audio) > expected_output_len:
                            output_audio = output_audio[:expected_output_len]
                        else:
                            # Pad with zeros if too short
                            padding = np.zeros(expected_output_len - len(output_audio))
                            output_audio = np.concatenate([output_audio, padding])
                    
                    output_audio = agc.process(output_audio * volume_level * volume)  # Apply AGC to output audio

                    # Stream output
                    audio_streamer.put_output_audio(output_audio)
                    
                    # Performance monitoring
                    if process_time > expectChunk:
                        print(f"Warning: Processing time {process_time:.3f}s exceeds chunk time {expectChunk}s")
                    
                except Exception as e:
                    print(f"LC_RVC processing error: {e}")
                    # Output silence on error
                    silence_output = np.zeros(int(expectChunk * output_sampling_rate))
                    audio_streamer.put_output_audio(silence_output)
                    continue
                
    except KeyboardInterrupt:
        print("\nStopping voice conversion...")
    finally:
        audio_streamer.stop()
        print("Voice conversion stopped.")

def list_audio_devices():
    """List available audio devices for input and output."""
    print("Available Audio Devices:")
    print("=" * 50)
    
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        device_type = []
        # Type ignore comments to suppress false positives
        if device['max_input_channels'] > 0:  # type: ignore
            device_type.append("INPUT")
        if device['max_output_channels'] > 0:  # type: ignore
            device_type.append("OUTPUT")
        
        print(f"[{i:2d}] {device['name']}")  # type: ignore
        print(f"     Type: {' & '.join(device_type)}")
        print(f"     Sample rates: {device['default_samplerate']:.0f}Hz")  # type: ignore
        print(f"     Channels: IN={device['max_input_channels']}, OUT={device['max_output_channels']}")  # type: ignore
        print()
    
    default_input = sd.default.device[0]
    default_output = sd.default.device[1]
    print(f"Default input device: [{default_input}] {sd.query_devices(default_input)['name']}")  # type: ignore
    print(f"Default output device: [{default_output}] {sd.query_devices(default_output)['name']}")  # type: ignore

def find_audio_device(device_name: str, device_type: str = "both") -> Optional[int]:
    """
    Find audio device index by name.
    
    Args:
        device_name (str): Name or partial name of the device to find
        device_type (str): Type of device to find ("input", "output", or "both")
    
    Returns:
        Optional[int]: Device index if found, None otherwise
    """
    devices = sd.query_devices()
    device_name_lower = device_name.lower()
    
    for i, device in enumerate(devices):
        if device_name_lower in str(device['name']).lower():  # type: ignore
            # Check if device supports the requested type
            if device_type == "input" and device['max_input_channels'] > 0:  # type: ignore
                return i
            elif device_type == "output" and device['max_output_channels'] > 0:  # type: ignore
                return i
            elif device_type == "both":
                return i
    
    return None


def select_audio_devices_interactive() -> tuple[Optional[int], Optional[int]]:
    """
    Interactive audio device selection.
    
    Returns:
        tuple[Optional[int], Optional[int]]: (input_device_index, output_device_index)
    """
    print("Available Audio Devices:")
    print("=" * 50)
    
    devices = sd.query_devices()
    input_devices = []
    output_devices = []
    
    for i, device in enumerate(devices):
        device_types = []
        if device['max_input_channels'] > 0:  # type: ignore
            device_types.append("INPUT")
            input_devices.append((i, str(device['name'])))  # type: ignore
        if device['max_output_channels'] > 0:  # type: ignore
            device_types.append("OUTPUT")
            output_devices.append((i, str(device['name'])))  # type: ignore
        
        print(f"[{i:2d}] {device['name']}")  # type: ignore
        print(f"     Type: {' & '.join(device_types)}")
        print(f"     Sample rates: {device['default_samplerate']:.0f}Hz")  # type: ignore
        print(f"     Channels: IN={device['max_input_channels']}, OUT={device['max_output_channels']}")  # type: ignore
        print()
    
    # Select input device
    print("Input Devices:")
    for i, (device_idx, device_name) in enumerate(input_devices):
        print(f"  {i}: [{device_idx}] {device_name}")
    
    try:
        input_choice = int(input("Select input device (number): "))
        input_device_index = input_devices[input_choice][0]
    except (ValueError, IndexError):
        print("Invalid input device selection, using default")
        input_device_index = None
    
    # Select output device
    print("\nOutput Devices:")
    for i, (device_idx, device_name) in enumerate(output_devices):
        print(f"  {i}: [{device_idx}] {device_name}")
    
    try:
        output_choice = int(input("Select output device (number): "))
        output_device_index = output_devices[output_choice][0]
    except (ValueError, IndexError):
        print("Invalid output device selection, using default")
        output_device_index = None
    
    return input_device_index, output_device_index
