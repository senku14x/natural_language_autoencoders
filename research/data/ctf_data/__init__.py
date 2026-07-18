"""Dataset generation for Counterfactual Difference NLA v1 (tokenizer-only stage).

Builds the frozen prompt-pair dataset + manifest that the GPU stage consumes.
No model weights are loaded anywhere in this package — tokenizer only.
"""

__version__ = "0.1.0"
