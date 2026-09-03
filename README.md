# Lab 6 — Transfer Learning y Fine-Tuning (ResNet-18 → CIFAR-10)

Deep Learning · Semana 9 · Universidad del Valle de Guatemala

Adaptación de **ResNet-18 preentrenada en ImageNet** a **CIFAR-10** con tres
estrategias de transfer learning de complejidad creciente, más un análisis
cuantitativo de las representaciones que aprende cada una.

Todo el trabajo vive en un solo archivo: **`S9 - Lab6_Semana9.ipynb`**.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Después, abrir el notebook y ejecutar **de arriba hacia abajo**. La primera
corrida descarga CIFAR-10 (~170 MB) a `data/`; ambas cosas están en `.gitignore`.

Las celdas 2 y 3 fijan `torch.manual_seed(42)`, así que una corrida lineal
completa es reproducible. Ejecutar celdas fuera de orden rompe esa garantía.

### CPU o GPU

`requirements.txt` fija las ruedas `+cpu`. Con ese setup el lab corre en pocos
minutos, porque usa un subconjunto de 2000 train / 500 val a 64×64 px.

Para GPU NVIDIA:

```bash
pip install -r requirements-gpu.txt
```

El notebook ya elige el dispositivo solo (`DEVICE = 'cuda' if
torch.cuda.is_available() else 'cpu'`, celda 2), así que no hay que tocar código.
Verificar con:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_arch_list())   # debe incluir la arquitectura de tu GPU
```

**Los resultados de abajo se generaron en GPU** (RTX 5070 Ti, `cu130`): las tres
estrategias completas toman 18 segundos, contra ~10 minutos en CPU.

Tres cosas que vale saber:

- **Blackwell (RTX 50xx) necesita `cu128` o superior.** Esas GPUs son compute
  capability 12.0 (`sm_120`) y los builds `cu126` no traen kernels para ellas:
  `torch.cuda.is_available()` da `True` pero el primer kernel falla con
  `no kernel image is available for execution on the device`. Por eso
  `requirements-gpu.txt` usa `cu130`. Confirmar con
  `'sm_120' in torch.cuda.get_arch_list()`.
- **En GPU el seed no alcanza para reproducir.** Sin configuración extra, dos
  corridas del Bloque 3 con el mismo seed dan resultados distintos (se midió
  78.60 % vs 78.80 %, y `sep_dlr` 0.7216 vs 0.7179). Los Bloques 1 y 2 sí salen
  idénticos: con el backbone congelado el backward se corta antes de los kernels
  de convolución. En el Bloque 3 se entrena la red completa y los kernels de
  `wgrad` de cuDNN acumulan con atomics, así que el orden de suma cambia entre
  corridas. La celda 3 lo arregla con `cudnn.deterministic`,
  `cudnn.benchmark = False` y `use_deterministic_algorithms`; el costo medido es
  nulo (17.8 s vs 18.4 s) y las corridas pasan a ser bit-idénticas.
- **Los números en CPU y en GPU no son los mismos.** El punto de partida sí es
  idéntico (la inicialización de pesos y el shuffle ocurren en CPU), pero el
  orden de acumulación en punto flotante difiere y se amplifica sobre 320 pasos.
  Feature extraction y fine-tuning parcial convergen a lo mismo en ambos
  dispositivos; el LR diferencial no (78.60 % en GPU, 78.20 % en CPU).

Si Jupyter no encuentra el `.venv`, registrarlo como kernel:

```bash
python -m ipykernel install --user --name lab6-venv --display-name "Python (lab6 .venv)"
```

---

## Estructura del notebook

| Bloque | Qué hace | Pts |
|---|---|---|
| 0 | Imports, datos, `train_epoch` / `eval_epoch` (dado) | — |
| 1 | **Feature extraction** — backbone congelado, solo `fc` entrena (5,130 params) | 20 |
| 2 | **Fine-tuning parcial** — descongelar `layer4` + `fc` (8.4 M params) | 20 |
| 3 | **Fine-tuning completo** con *learning rate* diferencial por grupo | 20 |
| 4 | **Análisis de representaciones** — embeddings de 512-d y ratio de separabilidad | 15 |
| 5 | Preguntas de análisis (1, 2, 3) | 25 |

### Learning rates diferenciales (Bloque 3)

| Grupo | `conv1`+`bn1` | `layer1` | `layer2` | `layer3` | `layer4` | `fc` |
|---|---|---|---|---|---|---|
| lr | 1e-6 | 1e-5 | 5e-5 | 1e-4 | 5e-4 | 1e-3 |

Los 11,181,642 parámetros del modelo quedan cubiertos por exactamente un grupo.

### Métrica del Bloque 4

Ratio de separabilidad (Fisher) sobre los embeddings de 512-d que salen de
`avgpool`, antes de `fc`:

$$\text{sep} = \frac{\sum_k n_k \lVert \mu_k - \mu \rVert^2}{\sum_k \sum_{i:\,y_i=k} \lVert h_i - \mu_k \rVert^2}$$

Mayor ratio = clases más separadas relativo a su dispersión interna.

---

## Resultados

Corrida lineal completa (kernel limpio, `torch.manual_seed(42)`, 10 épocas por
estrategia, GPU en modo determinista). Estos números son los que quedan
guardados en el notebook y se reproducen ejecutándolo de arriba hacia abajo.

| Estrategia | Params entrenables | Mejor val acc | Separabilidad |
|---|---:|---:|---:|
| Feature extraction | 5,130 | 62.40 % (ep 7) | 0.0580 |
| Fine-tuning parcial (`layer4`) | 8,398,858 | 74.80 % (ep 8) | 0.2124 |
| LR diferencial (full FT) | 11,181,642 | **78.60 %** (ep 7) | **0.7141** |

Verificación automática: **75/75 pts** de código.

Más capacidad adaptable ⇒ mejor accuracy y representaciones más separables,
a costa de más cómputo y más sobreajuste: con solo 2,000 imágenes de
entrenamiento, el fine-tuning parcial llega a 99.95 % de train accuracy y el
LR diferencial ve su `val_loss` subir de 0.761 (época 2) a 1.060 (época 10).
Feature extraction es el único que casi no sobreajusta — y también el que
tiene el techo más bajo.

Cuánto se movió cada capa respecto a los pesos de ImageNet en el modelo con LR
diferencial, medido como `||W_ft - W_0||_F / ||W_0||_F`
sobre `named_parameters()`:

| `conv1` | `layer1` | `layer2` | `layer3` | `layer4` |
|---:|---:|---:|---:|---:|
| 1.0e-04 | 2.8e-03 | 1.9e-02 | 5.4e-02 | 2.3e-01 |

Es decir: el stem queda intacto (0.010 %) y `layer4` se reescribe (23 %), que es
exactamente lo que buscan los learning rates diferenciales. Lo calcula la celda
que sigue a la verificación del Bloque 4, como evidencia para la Pregunta 3b.

> Ojo al medir esto: hay que usar `named_parameters()`, no `state_dict()`. Los
> buffers de BatchNorm (`running_mean`/`running_var`) se actualizan por media
> móvil en cada forward en modo `train()`, sin importar `requires_grad`. En el
> modelo de feature extraction, `layer4` cambia exactamente 0 en parámetros
> pero un 44 % en buffers.

---

## División del trabajo

| Persona | Bloques | Preguntas |
|---|---|---|
| Jose Ruiz | 1 | 1 (a, b, c) |
| Gerardo | 2 y 3 | — |
| Jose Auyón | 4 | 2 y 3 |
