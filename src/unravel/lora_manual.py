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

    # Detectar el device del modelo. Los wrappers que creemos arrancan en CPU;
    # al final movemos todo al device original para evitar mismatches.
    try:
        device = next(net.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

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
        elif isinstance(original, nn.LSTM):
            wrapper = LoraLSTM(original, r=r, alpha=alpha, dropout=dropout)
        else:
            raise ValueError(
                f"Módulo {name!r} es {type(original).__name__}, "
                f"se esperaba nn.Linear, nn.Conv2d o nn.LSTM"
            )
        _set_child(parent, attr_name, wrapper)

    # Movemos el modelo al device original (los wrappers nuevos se mueven con él).
    net = net.to(device)

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


class LoraLSTMCell(nn.Module):
    """LoRA wrapper para una celda LSTM (un step temporal, una dirección).

    A diferencia de LoraLinear y LoraConv2d, esta clase NO envuelve un módulo
    original — reimplementa el cómputo del cell manualmente, porque la
    corrección LoRA tiene que sumarse ANTES de las activaciones (sigmoid/tanh)
    que cierran cada gate.

    Recibe directamente los 4 tensores de pesos del LSTM original (weight_ih,
    weight_hh, bias_ih, bias_hh) y los guarda como buffers (congelados,
    se mueven con .to(device) pero no se actualizan). Aplica LoRA al tensor
    concatenado weight_ih (4*hidden × input) y al weight_hh (4*hidden × hidden),
    según la decisión B-2: una A compartida entre las 4 gates de cada set,
    pero la matriz B se rompe en 4 sub-bloques verticalmente, dándole a cada
    gate su propia corrección.

    Args:
        weight_ih: shape (4*hidden, input_size). Congelado.
        weight_hh: shape (4*hidden, hidden_size). Congelado.
        bias_ih:   shape (4*hidden,). Congelado.
        bias_hh:   shape (4*hidden,). Congelado.
        r: rango LoRA. Mismo para los dos bloques (ih y hh).
        alpha: factor de escala. Convención alpha = 2 * r.
        dropout: dropout aplicado al input y al estado anterior antes de
            entrar a la rama LoRA.

    Forward(x, (h_prev, c_prev)) -> (h_new, c_new), igual que nn.LSTMCell.
    """

    def __init__(
        self,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"r debe ser > 0, recibí {r}")

        # Pesos del LSTM original como nn.Parameter con requires_grad=False:
        # se mueven con .to(device), entran en state_dict, y SÍ se cuentan en
        # net.parameters() (necesario para que el % entrenable de stats sea
        # coherente con LoraLinear/LoraConv2d que envuelven módulos cuyos weights
        # ya son Parameters). No se actualizan porque tienen requires_grad=False.
        self.weight_ih = nn.Parameter(weight_ih.detach().clone(), requires_grad=False)
        self.weight_hh = nn.Parameter(weight_hh.detach().clone(), requires_grad=False)
        self.bias_ih = nn.Parameter(bias_ih.detach().clone(), requires_grad=False)
        self.bias_hh = nn.Parameter(bias_hh.detach().clone(), requires_grad=False)

        input_size = weight_ih.shape[1]
        hidden_size = weight_ih.shape[0] // 4

        # LoRA para weight_ih: A (input → r), B (r → 4*hidden).
        self.lora_ih_A = nn.Linear(input_size, r, bias=False)
        nn.init.kaiming_uniform_(self.lora_ih_A.weight, a=math.sqrt(5))
        self.lora_ih_B = nn.Linear(r, 4 * hidden_size, bias=False)
        nn.init.zeros_(self.lora_ih_B.weight)

        # LoRA para weight_hh: A (hidden → r), B (r → 4*hidden).
        self.lora_hh_A = nn.Linear(hidden_size, r, bias=False)
        nn.init.kaiming_uniform_(self.lora_hh_A.weight, a=math.sqrt(5))
        self.lora_hh_B = nn.Linear(r, 4 * hidden_size, bias=False)
        nn.init.zeros_(self.lora_hh_B.weight)

        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.r = r
        self.alpha = alpha

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state

        # Pre-activation gates: parte original (con pesos congelados).
        gates_orig = (
            nn.functional.linear(x, self.weight_ih, self.bias_ih)
            + nn.functional.linear(h_prev, self.weight_hh, self.bias_hh)
        )
        # Corrección LoRA: dropout → A → B (ih y hh) y se suma escalada.
        x_drop = self.lora_dropout(x)
        h_drop = self.lora_dropout(h_prev)
        gates_lora = (
            self.lora_ih_B(self.lora_ih_A(x_drop))
            + self.lora_hh_B(self.lora_hh_A(h_drop))
        )
        gates = gates_orig + self.scaling * gates_lora

        # Cuatro gates: input, forget, cell candidate, output.
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # Update memoria y output.
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"
        )


class LoraLSTM(nn.Module):
    """LSTM bidireccional multi-capa con LoRA aplicado a cada celda interna.

    Reemplaza un nn.LSTM original con un grid de LoraLSTMCell — una celda por
    cada (capa, dirección). El forward replica el comportamiento de nn.LSTM:
    para cada capa, las direcciones procesan la secuencia en paralelo, sus
    outputs se concatenan en la dimensión de features, y eso es el input de
    la próxima capa. Si el LSTM original tiene dropout entre capas, se aplica.

    forward(x, hidden=None) -> (output, (h_n, c_n)) — misma API que nn.LSTM.

    Args:
        original: nn.LSTM con los pesos preentrenados. Se respeta su input_size,
            hidden_size, num_layers, bidirectional, batch_first y dropout.
        r: rango LoRA, mismo para todas las celdas.
        alpha: factor de escala.
        dropout: dropout dentro de la rama LoRA (no es el dropout entre capas).
    """

    def __init__(
        self,
        original: nn.LSTM,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = original.input_size
        self.hidden_size = original.hidden_size
        self.num_layers = original.num_layers
        self.bidirectional = bool(original.bidirectional)
        self.batch_first = bool(original.batch_first)
        self.num_directions = 2 if self.bidirectional else 1
        self.layer_dropout = (
            nn.Dropout(original.dropout) if original.dropout > 0 else nn.Identity()
        )

        # Una celda por (capa, dirección), copiando los pesos del nn.LSTM original.
        self.cells = nn.ModuleList()
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                suffix = "_reverse" if direction == 1 else ""
                weight_ih = getattr(original, f"weight_ih_l{layer}{suffix}")
                weight_hh = getattr(original, f"weight_hh_l{layer}{suffix}")
                bias_ih = getattr(original, f"bias_ih_l{layer}{suffix}")
                bias_hh = getattr(original, f"bias_hh_l{layer}{suffix}")
                cell = LoraLSTMCell(
                    weight_ih, weight_hh, bias_ih, bias_hh,
                    r=r, alpha=alpha, dropout=dropout,
                )
                self.cells.append(cell)

        self.r = r
        self.alpha = alpha

    def _cell(self, layer: int, direction: int) -> LoraLSTMCell:
        return self.cells[layer * self.num_directions + direction]

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Internamente trabajamos en (batch, seq, features); convertimos si hace falta.
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch_size, seq_len, _ = x.shape

        if hidden is None:
            shape = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
            h0 = x.new_zeros(shape)
            c0 = x.new_zeros(shape)
        else:
            h0, c0 = hidden

        layer_input = x
        h_out_all: list[torch.Tensor] = []
        c_out_all: list[torch.Tensor] = []

        for layer in range(self.num_layers):
            outputs_per_dir: list[torch.Tensor] = []
            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction
                h = h0[idx]
                c = c0[idx]
                cell = self._cell(layer, direction)

                # Recorrer la secuencia en el orden correspondiente a la dirección.
                step_outputs: list[torch.Tensor] = []
                times = range(seq_len) if direction == 0 else range(seq_len - 1, -1, -1)
                for t in times:
                    h, c = cell(layer_input[:, t, :], (h, c))
                    step_outputs.append(h)
                if direction == 1:
                    step_outputs = list(reversed(step_outputs))
                seq_out = torch.stack(step_outputs, dim=1)
                outputs_per_dir.append(seq_out)
                h_out_all.append(h)
                c_out_all.append(c)

            # Concatenar las direcciones por la dimensión de features.
            layer_input = torch.cat(outputs_per_dir, dim=-1)
            # Dropout entre capas (igual que nn.LSTM, no se aplica después de la última).
            if layer < self.num_layers - 1:
                layer_input = self.layer_dropout(layer_input)

        output = layer_input
        h_n = torch.stack(h_out_all, dim=0)
        c_n = torch.stack(c_out_all, dim=0)

        if not self.batch_first:
            output = output.transpose(0, 1)
        return output, (h_n, c_n)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bidirectional={self.bidirectional}, "
            f"batch_first={self.batch_first}, r={self.r}, alpha={self.alpha}"
        )
