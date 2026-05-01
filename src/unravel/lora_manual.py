"""LoRA implementado manualmente, sin la librería PEFT.

Define `LoraLinear`, un wrapper que envuelve un `nn.Linear` original y le
suma una corrección de bajo rango entrenable. Los pesos originales quedan
congelados; solo entrenan las matrices A y B de LoRA.

La idea matemática:
    forward original:  y = W·x + b
    forward con LoRA:  y = W·x + b + (alpha/r) · B·A·x

Donde A se inicializa random (Kaiming) y B se inicializa en ceros, de
forma que en el step 0 el modelo es exactamente el original.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoraLinear(nn.Module):
    """Wrapper LoRA para una capa nn.Linear.

    Args:
        original: la capa nn.Linear preentrenada que queremos adaptar. Sus
            pesos se congelan en el constructor.
        r: rango de la descomposición de bajo rango. Típicamente 8.
        alpha: factor de escala. Convención: alpha = 2 * r.
        dropout: dropout aplicado a la entrada antes de la rama LoRA, como
            regularización. 0.0 desactiva el dropout.

    El forward retorna:  original(x) + (alpha / r) * B(A(dropout(x)))
    """

    def __init__(
        self,
        original: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"r debe ser > 0, recibí {r}")

        # Guardamos la capa original y congelamos sus pesos.
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        # A: in_features → r. Init Kaiming uniforme (mismo default que nn.Linear).
        self.lora_A = nn.Linear(in_features, r, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # B: r → out_features. Init en ceros: garantiza que LoRA arranca como identidad.
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

        # Escalado alpha/r y dropout opcional sobre la entrada.
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Guardamos hiperparámetros para inspección/checkpoint.
        self.r = r
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rama original (pesos congelados).
        out = self.original(x)
        # Rama LoRA: dropout → A → B → escalado.
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"


class LoraConv2d(nn.Module):
    """Wrapper LoRA para una capa nn.Conv2d.

    Args:
        original: la capa nn.Conv2d preentrenada que queremos adaptar. Sus
            pesos se congelan en el constructor.
        r: rango de la descomposición de bajo rango. Típicamente 8.
        alpha: factor de escala. Convención: alpha = 2 * r.
        dropout: dropout aplicado a la entrada antes de la rama LoRA, como
            regularización. 0.0 desactiva el dropout.

    Implementación: descomponemos la convolución original en dos convs
    consecutivas:
        Conv2d_A: in_channels → r,    kernel kxk (igual al original)
        Conv2d_B: r            → out, kernel 1x1

    Forward:  original(x) + (alpha/r) · Conv2d_B(Conv2d_A(dropout(x)))
    """

    def __init__(
        self,
        original: nn.Conv2d,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"r debe ser > 0, recibí {r}")
        if original.groups != 1:
            raise NotImplementedError(
                f"LoraConv2d aún no soporta groups>1 (recibí groups={original.groups})"
            )

        # Congelamos los pesos del original.
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False

        # Conv2d_A: respeta la geometría espacial del original (kernel, stride,
        # padding, dilation) pero comprime canales a r. Sin bias.
        self.lora_A = nn.Conv2d(
            in_channels=original.in_channels,
            out_channels=r,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            dilation=original.dilation,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # Conv2d_B: 1x1, expande de r a out_channels. Init en ceros.
        self.lora_B = nn.Conv2d(
            in_channels=r,
            out_channels=original.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.r = r
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rama original (pesos congelados).
        out = self.original(x)
        # Rama LoRA: dropout → A (espacial) → B (1x1) → escalado.
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"


# === Lógica de aplicación ===

DEFAULT_TARGET_MODULES: list[str] = ["top.fnl.1", "top.cnn.1"]


def apply_lora_manual(
    net: nn.Module,
    target_modules: list[str] | None = None,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> tuple[nn.Module, dict]:
    """Aplica LoRA manualmente sobre los módulos de net especificados por nombre.

    Recorre net, encuentra cada módulo cuyo path (dotted style) coincida con un
    nombre en target_modules, y lo reemplaza in-place por su versión LoRA:
    LoraLinear si era nn.Linear, LoraConv2d si era nn.Conv2d.

    El forward de los wrappers preserva el output original en step 0 (porque
    B se inicializa en ceros), así el modelo arranca idéntico al original.

    Args:
        net: el modelo a adaptar.
        target_modules: lista de paths a los módulos. Si None, usa
            DEFAULT_TARGET_MODULES (los mismos que toca el lora_setup.py con PEFT).
        r: rango LoRA, mismo para todos los targets.
        alpha: factor de escala.
        dropout: dropout en la rama LoRA.

    Returns:
        (net, stats) donde stats tiene trainable_params, total_params,
        percent_trainable, target_modules.
    """
    if target_modules is None:
        target_modules = list(DEFAULT_TARGET_MODULES)

    # Congelamos TODOS los parámetros del modelo antes de aplicar wrappers.
    # Los wrappers van a crear sus propios A y B (entrenables por default),
    # así que el net resultante solo tiene los LoRA como entrenables.
    for p in net.parameters():
        p.requires_grad = False

    for name in target_modules:
        parent, attr_name, original = _resolve_module(net, name)
        if isinstance(original, nn.Linear):
            wrapper = LoraLinear(original, r=r, alpha=alpha, dropout=dropout)
        elif isinstance(original, nn.Conv2d):
            wrapper = LoraConv2d(original, r=r, alpha=alpha, dropout=dropout)
        else:
            raise ValueError(
                f"Módulo {name!r} es {type(original).__name__}, "
                f"se esperaba nn.Linear o nn.Conv2d"
            )
        _set_child(parent, attr_name, wrapper)

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    stats = {
        "trainable_params": trainable,
        "total_params": total,
        "percent_trainable": 100.0 * trainable / total,
        "target_modules": list(target_modules),
    }
    return net, stats


def _resolve_module(
    root: nn.Module, dotted_path: str
) -> tuple[nn.Module, str, nn.Module]:
    """Navega un path tipo 'top.fnl.1' y devuelve (parent, attr_name, target).

    Soporta atributos por nombre (getattr) e índices numéricos en
    Sequential / ModuleList.
    """
    parts = dotted_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    attr = parts[-1]
    target = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
    return parent, attr, target


def _set_child(parent: nn.Module, attr_name: str, new_module: nn.Module) -> None:
    """Setea el submódulo, manejando el caso de índice numérico (Sequential)."""
    if attr_name.isdigit():
        parent[int(attr_name)] = new_module
    else:
        setattr(parent, attr_name, new_module)


def lora_state_dict(model: nn.Module) -> dict:
    """Extrae solo los pesos LoRA del modelo para checkpoint liviano.

    Filtra el state_dict completo del modelo dejando solo las claves que
    pertenecen a los wrappers LoraLinear/LoraConv2d (lora_A y lora_B).
    Útil para guardar checkpoints chicos: no necesitamos persistir los
    pesos congelados del modelo original (esos vienen del preentrenamiento
    IAM y no cambian durante el fine-tuning).

    Para cargar después: reconstruir el modelo con apply_lora_manual y
    hacer `model.load_state_dict(saved, strict=False)`. Los pesos no-LoRA
    se mantienen como estaban en el modelo (i.e., los del IAM preentrenado).
    """
    return {
        k: v
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
