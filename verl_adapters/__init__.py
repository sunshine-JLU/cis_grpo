"""Verl adapters for models without standard HuggingFace multimodal processors.

To use with verl, add ``verl_adapters/`` to your ``PYTHONPATH`` and register
the custom processor in ``verl.utils.tokenizer`` by adding to
``_CUSTOM_PROCESSOR_CLASSES``:

.. code-block:: python

    _CUSTOM_PROCESSOR_CLASSES["internvl_chat"] = "verl_adapters.InternVLProcessor"
"""

from .internvl_processor import InternVLProcessor

__all__ = ["InternVLProcessor"]
