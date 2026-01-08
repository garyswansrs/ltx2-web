"""
Preset Management System for LTX-2 WebUI
Handles saving, loading, and managing generation presets.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Default paths
PRESETS_DIR = Path("./presets")
PRESETS_FILE = PRESETS_DIR / "presets.json"
DEFAULT_PRESET_NAME = "default"


@dataclass
class GenerationPreset:
    """A preset containing all generation settings."""
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Pipeline settings
    pipeline_type: str = "distilled"
    checkpoint_path: str = ""
    distilled_lora_path: str = "None"
    spatial_upsampler_path: str = ""
    gemma_path: str = "./models/gemma"
    
    # Generation parameters
    prompt: str = ""
    negative_prompt: str = ""
    height: int = 1024
    width: int = 1536
    num_frames: int = 121
    frame_rate: float = 24.0
    num_inference_steps: int = 40
    cfg_guidance_scale: float = 4.0
    seed: int = -1
    enable_fp8: bool = True
    
    # Image conditioning
    image_strength: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationPreset":
        """Create from dictionary."""
        # Filter out any extra keys not in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


class PresetManager:
    """Manages generation presets - save, load, delete, set default."""
    
    def __init__(self, presets_dir: Optional[Path] = None):
        self.presets_dir = presets_dir or PRESETS_DIR
        self.presets_file = self.presets_dir / "presets.json"
        self._presets: Dict[str, GenerationPreset] = {}
        self._default_preset_name: str = DEFAULT_PRESET_NAME
        self._ensure_directories()
        self._load_presets()
    
    def _ensure_directories(self):
        """Create necessary directories."""
        self.presets_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_presets(self):
        """Load presets from file."""
        if self.presets_file.exists():
            try:
                with open(self.presets_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._default_preset_name = data.get("default_preset", DEFAULT_PRESET_NAME)
                    presets_data = data.get("presets", {})
                    for name, preset_data in presets_data.items():
                        self._presets[name] = GenerationPreset.from_dict(preset_data)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Failed to load presets: {e}")
                self._presets = {}
        
        # Ensure default preset exists
        if DEFAULT_PRESET_NAME not in self._presets:
            self._create_default_preset()
    
    def _create_default_preset(self):
        """Create the default preset with sensible defaults."""
        default = GenerationPreset(
            name=DEFAULT_PRESET_NAME,
            description="Default generation settings for LTX-2",
            pipeline_type="distilled",
            prompt="A beautiful cinematic shot of nature, high quality, 4K",
            negative_prompt="blurry, low quality, distorted, ugly",
            height=1024,
            width=1536,
            num_frames=121,
            frame_rate=24.0,
            num_inference_steps=40,
            cfg_guidance_scale=4.0,
            seed=-1,
            enable_fp8=True,
            image_strength=1.0,
        )
        self._presets[DEFAULT_PRESET_NAME] = default
        self._save_presets()
    
    def _save_presets(self):
        """Save presets to file."""
        data = {
            "default_preset": self._default_preset_name,
            "presets": {name: preset.to_dict() for name, preset in self._presets.items()}
        }
        with open(self.presets_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def list_presets(self) -> List[str]:
        """Get list of preset names."""
        return list(self._presets.keys())
    
    def get_preset(self, name: str) -> Optional[GenerationPreset]:
        """Get a preset by name."""
        return self._presets.get(name)
    
    def get_default_preset(self) -> GenerationPreset:
        """Get the default preset."""
        return self._presets.get(self._default_preset_name, self._presets.get(DEFAULT_PRESET_NAME))
    
    def get_default_preset_name(self) -> str:
        """Get the name of the default preset."""
        return self._default_preset_name
    
    def save_preset(self, preset: GenerationPreset, overwrite: bool = False) -> bool:
        """Save a preset. Returns True if successful."""
        if preset.name in self._presets and not overwrite:
            # Update existing preset
            preset.updated_at = datetime.now().isoformat()
            preset.created_at = self._presets[preset.name].created_at
        
        self._presets[preset.name] = preset
        self._save_presets()
        return True
    
    def delete_preset(self, name: str) -> bool:
        """Delete a preset. Cannot delete the default preset."""
        if name == DEFAULT_PRESET_NAME:
            return False
        
        if name in self._presets:
            del self._presets[name]
            # If we deleted the current default, reset to "default"
            if self._default_preset_name == name:
                self._default_preset_name = DEFAULT_PRESET_NAME
            self._save_presets()
            return True
        return False
    
    def set_default(self, name: str) -> bool:
        """Set a preset as the default."""
        if name in self._presets:
            self._default_preset_name = name
            self._save_presets()
            return True
        return False
    
    def create_preset_from_settings(
        self,
        name: str,
        description: str = "",
        pipeline_type: str = "distilled",
        checkpoint_path: str = "",
        distilled_lora_path: str = "None",
        spatial_upsampler_path: str = "",
        gemma_path: str = "./models/gemma",
        prompt: str = "",
        negative_prompt: str = "",
        height: int = 1024,
        width: int = 1536,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        cfg_guidance_scale: float = 4.0,
        seed: int = -1,
        enable_fp8: bool = True,
        image_strength: float = 1.0,
    ) -> GenerationPreset:
        """Create a preset from individual settings."""
        preset = GenerationPreset(
            name=name,
            description=description,
            pipeline_type=pipeline_type,
            checkpoint_path=checkpoint_path,
            distilled_lora_path=distilled_lora_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_path=gemma_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            cfg_guidance_scale=cfg_guidance_scale,
            seed=seed,
            enable_fp8=enable_fp8,
            image_strength=image_strength,
        )
        return preset


# Global preset manager instance
_preset_manager: Optional[PresetManager] = None


def get_preset_manager() -> PresetManager:
    """Get or create the global preset manager."""
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager()
    return _preset_manager
