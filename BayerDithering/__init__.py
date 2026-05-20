from .core import DitherConfig
from .dither import BayerDither
from .cpu import CPUProcessor
from .matrix import matrices, generate_bayer_matrix
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("PIL").setLevel(logging.WARNING)

try:
    from .gpu import GPUProcessor
except ImportError:
    GPUProcessor = None

__all__ = ["DitherConfig", "BayerDither", "CPUProcessor", "GPUProcessor", "matrices", "generate_bayer_matrix"]