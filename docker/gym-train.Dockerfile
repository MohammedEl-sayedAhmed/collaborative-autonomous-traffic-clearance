# v1.0.0 line -- training image: the gym dev image plus stable-baselines3 (PPO).
# Build the base first (`./run.sh gym-build`), then `./run.sh train-build`.
#
# torch comes from PyTorch's CPU-only index: the default PyPI wheel is the CUDA
# build (~2.5 GB) and we train on CPU here, so the cpu wheel is ~10x smaller.
FROM caatc-gym

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Pin gymnasium to the version the M1 env is built against (0.29.1) and let the
# resolver pick a stable-baselines3 that agrees, so SB3 cannot silently pull a
# gymnasium 1.x with a different API.
RUN pip install --no-cache-dir "stable-baselines3>=2.4,<3" "gymnasium==0.29.1"

# pytest lives here too, so the SB3-dependent tests in caatc/tests/test_train.py
# actually run (they skip in the lighter test image).
RUN pip install --no-cache-dir pytest

CMD ["python", "-m", "caatc.train", "--help"]
