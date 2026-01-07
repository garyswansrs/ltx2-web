from dataclasses import replace
import gc
import logging

import torch

from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoder,
    AudioDecoderConfigurator,
    Vocoder,
    VocoderConfigurator,
)
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
    UPCAST_DURING_INFERENCE,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.upsampler import LatentUpsampler, LatentUpsamplerConfigurator
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.text_encoders.gemma import (
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaTextEncoderModel,
    AVGemmaTextEncoderModelConfigurator,
    module_ops_from_gemma_root,
)

logger = logging.getLogger(__name__)


class ModelLedger:
    """
    Central coordinator for loading and building models used in an LTX pipeline.
    The ledger wires together multiple model builders (transformer, video VAE encoder/decoder,
    audio VAE decoder, vocoder, text encoder, and optional latent upsampler) and exposes
    factory methods for constructing model instances.
    
    ### Model Caching (VRAM Persistence)
    
    By default, models are **cached in VRAM** after first creation. This dramatically
    speeds up batch processing and subsequent generations by avoiding model reload overhead.
    
    - Use ``enable_caching=True`` (default) to keep models in VRAM between calls
    - Use ``clear_cache()`` to manually free VRAM when switching configurations
    - Use ``clear_cache(model_name)`` to clear a specific model (e.g., "transformer")
    - Set ``enable_caching=False`` to disable caching (original behavior)
    
    ### Model Building
    Each model method (e.g. :meth:`transformer`, :meth:`video_decoder`, :meth:`text_encoder`)
    returns a cached model instance if available, or constructs a new one. The builder uses the
    :class:`~ltx_core.loader.registry.Registry` to load weights from the checkpoint,
    instantiates the model with the configured ``dtype``, and moves it to ``self.device``.
    
    ### Constructor parameters
    dtype:
        Torch dtype used when constructing all models (e.g. ``torch.bfloat16``).
    device:
        Target device to which models are moved after construction (e.g. ``torch.device("cuda")``).
    checkpoint_path:
        Path to a checkpoint directory or file containing the core model weights
        (transformer, video VAE, audio VAE, text encoder, vocoder). If ``None``, the
        corresponding builders are not created and calling those methods will raise
        a :class:`ValueError`.
    gemma_root_path:
        Base path to Gemma-compatible CLIP/text encoder weights. Required to
        initialize the text encoder builder; if omitted, :meth:`text_encoder` cannot be used.
    spatial_upsampler_path:
        Optional path to a latent upsampler checkpoint. If provided, the
        :meth:`spatial_upsampler` method becomes available; otherwise calling it raises
        a :class:`ValueError`.
    loras:
        Optional collection of LoRA configurations (paths, strengths, and key operations)
        that are applied on top of the base transformer weights when building the model.
    registry:
        Optional :class:`Registry` instance for weight caching across builders.
        Defaults to :class:`DummyRegistry` which performs no cross-builder caching.
    fp8transformer:
        If ``True``, builds the transformer with FP8 quantization and upcasting during inference.
    enable_caching:
        If ``True`` (default), models are cached in VRAM after first creation for faster
        subsequent calls. Set to ``False`` to recreate models on every call (original behavior).
    
    ### Creating Variants
    Use :meth:`with_loras` to create a new ``ModelLedger`` instance that includes
    additional LoRA configurations while sharing the same registry for weight caching.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        checkpoint_path: str | None = None,
        gemma_root_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras: LoraPathStrengthAndSDOps | None = None,
        registry: Registry | None = None,
        fp8transformer: bool = False,
        enable_caching: bool = True,
    ):
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.gemma_root_path = gemma_root_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.loras = loras or ()
        self.registry = registry or DummyRegistry()
        self.fp8transformer = fp8transformer
        self.enable_caching = enable_caching
        
        # Model cache - keeps models in VRAM for faster subsequent calls
        self._cached_transformer: X0Model | None = None
        self._cached_video_decoder: VideoDecoder | None = None
        self._cached_video_encoder: VideoEncoder | None = None
        self._cached_text_encoder: AVGemmaTextEncoderModel | None = None
        self._cached_audio_decoder: AudioDecoder | None = None
        self._cached_vocoder: Vocoder | None = None
        self._cached_spatial_upsampler: LatentUpsampler | None = None
        
        self.build_model_builders()

    def build_model_builders(self) -> None:
        if self.checkpoint_path is not None:
            self.transformer_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=LTXModelConfigurator,
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
                loras=tuple(self.loras),
                registry=self.registry,
            )

            self.vae_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vae_encoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.audio_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=AudioDecoderConfigurator,
                model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vocoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VocoderConfigurator,
                model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            if self.gemma_root_path is not None:
                self.text_encoder_builder = Builder(
                    model_path=self.checkpoint_path,
                    model_class_configurator=AVGemmaTextEncoderModelConfigurator,
                    model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                    registry=self.registry,
                    module_ops=module_ops_from_gemma_root(self.gemma_root_path),
                )

        if self.spatial_upsampler_path is not None:
            self.upsampler_builder = Builder(
                model_path=self.spatial_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=self.registry,
            )

    def _target_device(self) -> torch.device:
        if isinstance(self.registry, DummyRegistry) or self.registry is None:
            return self.device
        else:
            return torch.device("cpu")

    def with_loras(self, loras: LoraPathStrengthAndSDOps) -> "ModelLedger":
        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            gemma_root_path=self.gemma_root_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            loras=(*self.loras, *loras),
            registry=self.registry,
            fp8transformer=self.fp8transformer,
            enable_caching=self.enable_caching,
        )

    def clear_cache(self, model_name: str | None = None) -> None:
        """
        Clear cached models from VRAM.
        
        Args:
            model_name: Optional specific model to clear. One of: "transformer", 
                       "video_decoder", "video_encoder", "text_encoder", 
                       "audio_decoder", "vocoder", "spatial_upsampler".
                       If None, clears all cached models.
        """
        if model_name is None:
            # Clear all cached models
            models_to_clear = [
                "_cached_transformer",
                "_cached_video_decoder", 
                "_cached_video_encoder",
                "_cached_text_encoder",
                "_cached_audio_decoder",
                "_cached_vocoder",
                "_cached_spatial_upsampler",
            ]
            for attr in models_to_clear:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    logger.debug(f"Clearing cached {attr}")
                    setattr(self, attr, None)
        else:
            attr = f"_cached_{model_name}"
            if hasattr(self, attr):
                logger.debug(f"Clearing cached {model_name}")
                setattr(self, attr, None)
            else:
                raise ValueError(f"Unknown model name: {model_name}")
        
        # Free VRAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def transformer(self) -> X0Model:
        if not hasattr(self, "transformer_builder"):
            raise ValueError(
                "Transformer not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )
        
        # Return cached model if available
        if self.enable_caching and self._cached_transformer is not None:
            logger.debug("Using cached transformer (VRAM)")
            return self._cached_transformer
        
        logger.debug("Building transformer model...")
        if self.fp8transformer:
            fp8_builder = replace(
                self.transformer_builder,
                module_ops=(UPCAST_DURING_INFERENCE,),
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
            )
            model = X0Model(fp8_builder.build(device=self._target_device())).to(self.device).eval()
        else:
            model = (
                X0Model(self.transformer_builder.build(device=self._target_device(), dtype=self.dtype))
                .to(self.device)
                .eval()
            )
        
        # Cache the model
        if self.enable_caching:
            self._cached_transformer = model
            logger.debug("Transformer cached in VRAM")
        
        return model

    def video_decoder(self) -> VideoDecoder:
        if not hasattr(self, "vae_decoder_builder"):
            raise ValueError(
                "Video decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        # Return cached model if available
        if self.enable_caching and self._cached_video_decoder is not None:
            logger.debug("Using cached video_decoder (VRAM)")
            return self._cached_video_decoder

        logger.debug("Building video_decoder model...")
        model = self.vae_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        
        # Cache the model
        if self.enable_caching:
            self._cached_video_decoder = model
            logger.debug("video_decoder cached in VRAM")
        
        return model

    def video_encoder(self) -> VideoEncoder:
        if not hasattr(self, "vae_encoder_builder"):
            raise ValueError(
                "Video encoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        # Return cached model if available
        if self.enable_caching and self._cached_video_encoder is not None:
            logger.debug("Using cached video_encoder (VRAM)")
            return self._cached_video_encoder

        logger.debug("Building video_encoder model...")
        model = self.vae_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        
        # Cache the model
        if self.enable_caching:
            self._cached_video_encoder = model
            logger.debug("video_encoder cached in VRAM")
        
        return model

    def text_encoder(self) -> AVGemmaTextEncoderModel:
        if not hasattr(self, "text_encoder_builder"):
            raise ValueError(
                "Text encoder not initialized. Please provide a checkpoint path and gemma root path to the "
                "ModelLedger constructor."
            )

        # Return cached model if available
        if self.enable_caching and self._cached_text_encoder is not None:
            logger.debug("Using cached text_encoder (VRAM)")
            return self._cached_text_encoder

        logger.debug("Building text_encoder model...")
        model = self.text_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        
        # Cache the model
        if self.enable_caching:
            self._cached_text_encoder = model
            logger.debug("text_encoder cached in VRAM")
        
        return model

    def audio_decoder(self) -> AudioDecoder:
        if not hasattr(self, "audio_decoder_builder"):
            raise ValueError(
                "Audio decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        # Return cached model if available
        if self.enable_caching and self._cached_audio_decoder is not None:
            logger.debug("Using cached audio_decoder (VRAM)")
            return self._cached_audio_decoder

        logger.debug("Building audio_decoder model...")
        model = self.audio_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        
        # Cache the model
        if self.enable_caching:
            self._cached_audio_decoder = model
            logger.debug("audio_decoder cached in VRAM")
        
        return model

    def vocoder(self) -> Vocoder:
        if not hasattr(self, "vocoder_builder"):
            raise ValueError(
                "Vocoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        # Return cached model if available
        if self.enable_caching and self._cached_vocoder is not None:
            logger.debug("Using cached vocoder (VRAM)")
            return self._cached_vocoder

        logger.debug("Building vocoder model...")
        model = self.vocoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        
        # Cache the model
        if self.enable_caching:
            self._cached_vocoder = model
            logger.debug("vocoder cached in VRAM")
        
        return model

    def spatial_upsampler(self) -> LatentUpsampler:
        if not hasattr(self, "upsampler_builder"):
            raise ValueError("Upsampler not initialized. Please provide upsampler path to the ModelLedger constructor.")

        # Return cached model if available
        if self.enable_caching and self._cached_spatial_upsampler is not None:
            logger.debug("Using cached spatial_upsampler (VRAM)")
            return self._cached_spatial_upsampler

        logger.debug("Building spatial_upsampler model...")
        model = self.upsampler_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        
        # Cache the model
        if self.enable_caching:
            self._cached_spatial_upsampler = model
            logger.debug("spatial_upsampler cached in VRAM")
        
        return model
