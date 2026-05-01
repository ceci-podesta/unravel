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


def test_apply_lora_manual_reemplaza_y_preserva_forward():
    """En un modelo simple, apply_lora_manual debe:
    1. Reemplazar los módulos target con wrappers LoRA del tipo correcto.
    2. Preservar el forward del modelo en step 0 (B init en ceros).
    3. Devolver stats coherentes con la suma esperada.
    """
    import torch
    import torch.nn as nn
    from unravel.lora_manual import LoraConv2d, LoraLinear, apply_lora_manual

    torch.manual_seed(42)

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))  # global avg pool
            return self.fc(x)

    net = TinyNet()
    x = torch.randn(2, 3, 8, 8)
    y_before = net(x).detach().clone()

    net, stats = apply_lora_manual(net, target_modules=["conv", "fc"], r=8, alpha=16)

    assert isinstance(net.conv, LoraConv2d)
    assert isinstance(net.fc, LoraLinear)

    y_after = net(x)
    assert torch.allclose(y_before, y_after), (
        "El forward del modelo cambió tras aplicar LoRA en init"
    )

    expected_lora_conv = 8 * (3 * 3 * 3) + (8 * 8)  # 216 + 64
    expected_lora_fc = 8 * (8 + 4)  # 96
    assert stats["trainable_params"] == expected_lora_conv + expected_lora_fc


def test_apply_lora_manual_sobre_htrnet():
    """Aplicar sobre HTRNet con los targets default debe reemplazar
    top.fnl[1] y top.cnn[1] con wrappers LoRA, dejando el resto intacto.
    Valida que el path resolution funciona en árboles anidados reales.
    """
    from unravel.htr_model import HTRNet, default_arch_cfg
    from unravel.lora_manual import LoraConv2d, LoraLinear, apply_lora_manual

    nclasses = 80
    net = HTRNet(default_arch_cfg(), nclasses)
    net, stats = apply_lora_manual(net, r=8, alpha=16)  # default targets

    # Targets reemplazados.
    assert isinstance(net.top.fnl[1], LoraLinear)
    assert isinstance(net.top.cnn[1], LoraConv2d)

    # Resto intacto (sample check).
    assert isinstance(net.top.fnl[0], nn.Dropout)
    assert isinstance(net.top.rec, nn.LSTM)

    # % entrenable razonable: estamos tocando solo dos cabezas, debería ser bajo.
    assert 0.0 < stats["percent_trainable"] < 5.0, (
        f"% entrenable inesperado: {stats['percent_trainable']:.2f}%"
    )


def test_apply_lora_manual_falla_si_target_no_es_linear_ni_conv2d():
    """Si el target apunta a otro tipo de módulo (LSTM, etc.), debe ValueError."""
    import pytest
    from unravel.lora_manual import apply_lora_manual

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(8, 8)

        def forward(self, x):
            return self.rnn(x)

    net = TinyNet()
    with pytest.raises(ValueError, match="se esperaba nn.Linear o nn.Conv2d"):
        apply_lora_manual(net, target_modules=["rnn"], r=8)


def test_lora_state_dict_extrae_solo_pesos_lora():
    """lora_state_dict debe devolver solo las claves de los pesos LoRA
    (lora_A.weight y lora_B.weight de cada wrapper), no las de las capas
    originales congeladas ni otros parámetros del modelo.
    """
    from unravel.lora_manual import apply_lora_manual, lora_state_dict

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            x = self.conv(x).mean(dim=(2, 3))
            return self.fc(x)

    net = TinyNet()
    net, _ = apply_lora_manual(net, target_modules=["conv", "fc"], r=8, alpha=16)

    state = lora_state_dict(net)

    # 4 claves esperadas: lora_A.weight + lora_B.weight para cada wrapper.
    assert len(state) == 4, (
        f"Esperaba 4 claves LoRA, hay {len(state)}: {list(state.keys())}"
    )
    # Todas las claves contienen lora_A o lora_B.
    for k in state:
        assert "lora_A" in k or "lora_B" in k, f"Clave inesperada: {k}"
    # Ninguna corresponde a la capa original.
    for k in state:
        assert "original" not in k, f"Clave de original aparece en state: {k}"
