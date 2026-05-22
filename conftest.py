"""pytest configuration: force CPU-only mode for unit tests.

reward_model.py uses a module-level `device = 'cuda'` constant.  Tests patch
it to 'cpu' so they run without a CUDA-capable GPU.
"""
import reward_model as _rm

_rm.device = "cpu"
