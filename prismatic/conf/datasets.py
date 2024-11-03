"""
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: Tuple[Path, Path]       # Path to annotation file and images directory for `align` stage
    finetune_stage_components: Tuple[Path, Path]    # Path to annotation file and images directory for `finetune` stage

    dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
    # fmt: on


# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_mix665k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")


# [Multimodal-Only] LLava-v15 WITHOUT the Language-Only ShareGPT Data (No Co-Training)
@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    dataset_id: str = "llava-multimodal"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_stripped625k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V
@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LRV-Instruct
@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# Add by ZSXM
@dataclass
class INT_PI_PV3_2406110_Config(DatasetConfig):
    dataset_id: str = "int_pi_pv3_2406110"

    align_stage_components: Tuple[Path, Path] = (
        Path("playground/neurips/pretrain/2404181_quilt1m_pathcapdiff_pathinstructP1.json"),
        Path("datasets/combination/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("playground/neurips/finetune/2406110_int_patho-instruct_patho-vision-3.json"),
        Path("datasets/combination/"),
    )
    dataset_root_dir: Path = Path("/c22073/codes/llava-1.5/")

@dataclass
class PI_PV3_2406231_Config(DatasetConfig):
    dataset_id: str = "pi_pv3_2406231"

    align_stage_components: Tuple[Path, Path] = (
        Path("playground/neurips/pretrain/2404181_quilt1m_pathcapdiff_pathinstructP1.json"),
        Path("datasets/combination/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("playground/neurips/finetune/2406231_patho-instruct_patho-vision-3.json"),
        Path("datasets/combination/"),
    )
    dataset_root_dir: Path = Path("/c22073/codes/llava-1.5/")

@dataclass
class PI_PV3_2407091_Config(DatasetConfig):
    dataset_id: str = "pi_pv3_2407091"

    align_stage_components: Tuple[Path, Path] = (
        Path("playground/neurips/pretrain/2404181_quilt1m_pathcapdiff_pathinstructP1.json"),
        Path("datasets/combination/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("playground/neurips/finetune/2407091_patho-instruct_patho-vision-3.json"),
        Path("datasets/combination/"),
    )
    dataset_root_dir: Path = Path("/c22073/codes/llava-1.5/")

@dataclass
class PI_PV3_2408051_Config(DatasetConfig):
    dataset_id: str = "pi_pv3_2408051"

    align_stage_components: Tuple[Path, Path] = (
        Path("playground/neurips/pretrain/2404181_quilt1m_pathcapdiff_pathinstructP1.json"),
        Path("datasets/combination/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("playground/neurips/finetune/2408051_patho-instruct_patho-vision-3.json"),
        Path("datasets/combination/"),
    )
    dataset_root_dir: Path = Path("/c22073/codes/llava-1.5/")

@dataclass
class PI_PV3_2408231_Config(DatasetConfig):
    dataset_id: str = "pi_pv3_2408231"

    align_stage_components: Tuple[Path, Path] = (
        Path("playground/neurips/pretrain/2404181_quilt1m_pathcapdiff_pathinstructP1.json"),
        Path("datasets/combination/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("playground/neurips/finetune/2408231_patho-instruct_patho-vision-2.json"),
        Path("datasets/combination/"),
    )
    dataset_root_dir: Path = Path("/c22073/codes/llava-1.5/")

# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config

    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config

    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config

    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    # Add by ZSXM
    INT_PI_PV3_2406110 = INT_PI_PV3_2406110_Config
    PI_PV3_2406231 = PI_PV3_2406231_Config
    PI_PV3_2407091 = PI_PV3_2407091_Config
    PI_PV3_2408051 = PI_PV3_2408051_Config
    PI_PV3_2408231 = PI_PV3_2408231_Config

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
