# v1.0.0 line -- test image: the gym dev image plus pytest, for the unit tests
# and the ClearanceEnv headroom gate. Build the base first (`./run.sh gym-build`).
FROM caatc-gym
RUN pip install --no-cache-dir pytest
CMD ["python", "-m", "pytest", "-q", "caatc/tests"]
