"""
Auto-imported by Python at interpreter startup (via PYTHONPATH pointing at this directory) --
CI-only, never touched for local development runs.

GitHub-hosted macOS Apple Silicon runners report torch.backends.mps.is_available() == True, but
real MPS allocation fails immediately with "RuntimeError: MPS backend out of memory (MPS
allocated: 0 bytes...)" on the very first tensor allocation -- consistent with these runners'
Metal GPU access not actually being usable in whatever virtualization layer GitHub runs them in
(the API reports availability; the shared memory pool behind it does not actually work). Confirmed
by watching a real CI run: the "Confirm MPS is available" step passes, then every test using
`device = 'mps' if torch.backends.mps.is_available() else 'cpu'` (this port's standard pattern)
crashes on its first real tensor op.

Forcing MPS "unavailable" here makes that same existing fallback take the CPU path instead of
crashing, so CI verifies correctness on whatever compute is actually usable in that environment.
Only active when CI_FORCE_CPU=1 is set (tests.yml sets it) -- local runs on real Apple Silicon
hardware are completely unaffected and keep exercising MPS for real, which is what this port is
actually about; this file exists purely to get a truthful, non-crashing CI signal despite that
runner-level limitation, not to claim MPS was ever really tested there.
"""
import os
if os.environ.get('CI_FORCE_CPU') == '1':
    import torch
    torch.backends.mps.is_available = lambda: False
