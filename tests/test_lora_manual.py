"""Tests para LoraLinear (LoRA implementado manualmente sin PEFT)."""
import torch
import torch.nn as nn

from unravel.lora_manual import LoraConv2d, LoraLinear


def test_lora_linear_init_iguala_al_original():
    """En el step 0 (recién creada, sin entrenar), LoraLinear debe dar
    exactamente el mismo output que el nn.Linear original que envuelve.

    Esto valida la propiedad clave de la inicialización: B se inicializa
    en ceros, entonces B·A·x = 0 y el forward total = original(x) + 0.
    Sin esta propiedad, el modelo arrancaría perturbado respecto al
    preentrenado y perderíamos parte del valor del fine-tuning.
    """
    torch.manual_seed(42)

    # 1. Una capa nn.Linear con pesos random (simula una capa preentrenada).
    original = nn.Linear(in_features=20, out_features=10)

    # 2. Un tensor de input arbitrario: batch de 4 vectores de 20 features.
    x = torch.randn(4, 20)

    # 3. Salida del original.
    y_orig = original(x)

    # 4. Envolver el original con LoRA y calcular su salida.
    lora_layer = LoraLinear(original, r=8, alpha=16)
    y_lora = lora_layer(x)

    # 5. Las dos salidas tienen que ser iguales bit-a-bit (no es aproximación,
    #    porque 0 * cualquier_cosa = 0 exacto, sin error numérico).
    diff_max = (y_orig - y_lora).abs().max().item()
    assert torch.allclose(y_orig, y_lora), (
        f"LoraLinear recién creada debería dar el mismo output que el "
        f"original, pero la diferencia máxima fue {diff_max}"
    )


def test_lora_linear_congela_pesos_del_original():
    """Los pesos del nn.Linear original deben quedar con requires_grad=False
    al ser envueltos en LoraLinear. Sin esto, el optimizer también los
    actualizaría y se rompería la lógica de LoRA (entrenar solo A y B)."""
    original = nn.Linear(20, 10)
    # Antes de envolver, requires_grad es True (default de nn.Linear).
    assert original.weight.requires_grad is True
    assert original.bias.requires_grad is True

    lora_layer = LoraLinear(original, r=8, alpha=16)

    # Después de envolver, los pesos del original deben estar congelados.
    assert lora_layer.original.weight.requires_grad is False, (
        "Los pesos del original no se congelaron"
    )
    assert lora_layer.original.bias.requires_grad is False, (
        "El bias del original no se congeló"
    )

    # Y los pesos de A y B sí deben entrenar.
    assert lora_layer.lora_A.weight.requires_grad is True
    assert lora_layer.lora_B.weight.requires_grad is True


def test_lora_linear_cantidad_de_parametros_entrenables():
    """Los parámetros entrenables deben ser exactamente r·(in + out).
    Para LoraLinear(20 → 10) con r=8: 8·(20+10) = 240.

    A: r × in = 8 × 20 = 160
    B: out × r = 10 × 8 = 80
    Total: 240 entrenables.
    """
    original = nn.Linear(in_features=20, out_features=10)
    lora_layer = LoraLinear(original, r=8, alpha=16)

    trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    expected = 8 * (20 + 10)

    assert trainable == expected, (
        f"Esperaba {expected} parámetros entrenables, pero hay {trainable}"
    )


def test_lora_conv2d_init_iguala_al_original():
    """En el step 0, LoraConv2d debe dar exactamente el mismo output que el
    nn.Conv2d original. Misma propiedad clave que LoraLinear: B init en
    ceros → rama LoRA produce 0 → forward total = original(x) + 0.
    """
    torch.manual_seed(42)
    original = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
    # Input 4D: batch=2, channels=4, H=W=8.
    x = torch.randn(2, 4, 8, 8)

    y_orig = original(x)

    lora_layer = LoraConv2d(original, r=8, alpha=16)
    y_lora = lora_layer(x)

    diff_max = (y_orig - y_lora).abs().max().item()
    assert torch.allclose(y_orig, y_lora), (
        f"LoraConv2d recién creada debería dar el mismo output que el "
        f"original, pero la diferencia máxima fue {diff_max}"
    )


def test_lora_conv2d_congela_pesos_del_original():
    """Los pesos del Conv2d original deben quedar congelados; los de A y B
    deben entrenar.
    """
    original = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
    assert original.weight.requires_grad is True
    assert original.bias.requires_grad is True

    lora_layer = LoraConv2d(original, r=8, alpha=16)

    assert lora_layer.original.weight.requires_grad is False, (
        "Los pesos del Conv2d original no se congelaron"
    )
    assert lora_layer.original.bias.requires_grad is False, (
        "El bias del Conv2d original no se congeló"
    )
    assert lora_layer.lora_A.weight.requires_grad is True
    assert lora_layer.lora_B.weight.requires_grad is True


def test_lora_conv2d_cantidad_de_parametros_entrenables():
    """Para LoraConv2d:
        A: r · in_channels · kH · kW
        B: out_channels · r · 1 · 1

    Para Conv2d(4 → 8, kernel 3x3) con r=8:
        A: 8 · 4 · 3 · 3 = 288
        B: 8 · 8 · 1 · 1 = 64
        Total: 352
    """
    original = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
    lora_layer = LoraConv2d(original, r=8, alpha=16)

    trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    expected = 8 * (4 * 3 * 3) + (8 * 8 * 1 * 1)  # = 288 + 64 = 352

    assert trainable == expected, (
        f"Esperaba {expected} parámetros entrenables, pero hay {trainable}"
    )
