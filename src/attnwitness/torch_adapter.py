"""Optional, explicit CPU torch boundary; no hooks or tensor mutation."""
import torch
from .audit import audit
from .contracts import ContractError, positive_int


def _array(tensor, name, budget):
    if tensor is None:
        return None
    if type(tensor) is not torch.Tensor:
        raise ContractError(f'{name} must be a plain torch.Tensor; subclasses are not modeled')
    if tensor.device.type != 'cpu' or tensor.layout != torch.strided:
        raise ContractError(f'{name} must be a dense CPU tensor')
    if tensor.numel() > budget:
        raise ContractError(f'{name} exceeds the cell budget before conversion')
    if tensor.dtype not in (torch.bool, torch.float16, torch.float32, torch.float64):
        raise ContractError(f'{name} has an unsupported mask dtype')
    try:
        return tensor.detach().numpy()
    except (TypeError, RuntimeError) as exc:
        raise ContractError(f'{name} cannot be represented as a NumPy view') from exc


def audit_torch(mask, policy, **kwargs):
    """Audit a supported CPU tensor; finite biases do not become hard blocks."""
    maximum = positive_int(kwargs.get('max_cells', 1048576), 'max_cells')
    if 'key_padding_mask' in kwargs:
        kwargs['key_padding_mask'] = _array(kwargs['key_padding_mask'], 'key_padding_mask', maximum)
    return audit(_array(mask, 'mask', maximum), policy, **kwargs)
