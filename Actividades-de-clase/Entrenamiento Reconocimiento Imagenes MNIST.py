# -*- coding: utf-8 -*-
"""Reconocimiento de dígitos MNIST con PyTorch y Streamlit.

Un solo archivo ejecutable. Incluye:
- Arquitectura MLP adaptada del código original: 784 -> 64 -> 1024 -> 10.
- Entrenamiento con validación cruzada estratificada de 5 particiones.
- Selección del mejor modelo usando exclusivamente validación.
- Evaluación final, una sola vez, sobre el test oficial de MNIST.
- Procesamiento de imágenes externas y aplicación Streamlit en Localhost.

Ejecución:
    streamlit run app.py
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# 1. Configuración general y rutas
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "resultados"
MODEL_PATH = MODEL_DIR / "mejor_modelo_mnist.pth"
CV_PATH = RESULTS_DIR / "validacion_cruzada.csv"
TEST_PATH = RESULTS_DIR / "evaluacion_test.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_SPLITS = 5
NUM_CLASSES = 10
IMAGE_SIZE = 28

# -----------------------------------------------------------------------------
# 2. Arquitectura: conserva las capas del perceptrón original
# -----------------------------------------------------------------------------
class DigitMLP(nn.Module):
    """Red neuronal densa: 784 -> 64 -> 1024 -> 10."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28):
            raise ValueError(
                f"Entrada esperada [lote, 1, 28, 28]; recibida {tuple(x.shape)}"
            )
        return self.network(x)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 3. Preprocesamiento de imágenes externas
# -----------------------------------------------------------------------------
def otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    levels = np.arange(256, dtype=np.float64)
    cumulative_count = np.cumsum(histogram)
    cumulative_sum = np.cumsum(histogram * levels)
    total_sum = cumulative_sum[-1]
    weight_background = cumulative_count
    weight_foreground = total - weight_background
    valid = (weight_background > 0) & (weight_foreground > 0)
    mean_background = np.divide(
        cumulative_sum, weight_background,
        out=np.zeros_like(cumulative_sum), where=weight_background > 0
    )
    mean_foreground = np.divide(
        total_sum - cumulative_sum, weight_foreground,
        out=np.zeros_like(cumulative_sum), where=weight_foreground > 0
    )
    variance = weight_background * weight_foreground * (
        mean_background - mean_foreground
    ) ** 2
    variance[~valid] = -1
    return int(np.argmax(variance))


def shift_image(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = array.shape
    shifted = np.zeros_like(array)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    dx0, dx1 = max(0, dx), min(w, w + dx)
    dy0, dy1 = max(0, dy), min(h, h + dy)
    if sx1 > sx0 and sy1 > sy0:
        shifted[dy0:dy1, dx0:dx1] = array[sy0:sy1, sx0:sx1]
    return shifted


def preprocess_external_image(image: Image.Image) -> tuple[torch.Tensor, Image.Image]:
    """Convierte tinta negra sobre papel blanco o trazo blanco/fondo negro
    al dominio MNIST: fondo negro=0 y trazo claro=1.
    """
    if image is None:
        raise ValueError("No se recibió ninguna imagen.")
    gray_image = ImageOps.exif_transpose(image).convert("L")
    if gray_image.width < 5 or gray_image.height < 5:
        raise ValueError("La imagen es demasiado pequeña.")

    gray = np.asarray(gray_image, dtype=np.uint8)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    # Si los bordes son claros, se asume papel blanco y se invierte la imagen.
    if float(np.median(border)) > 127:
        gray = 255 - gray

    gray = np.asarray(ImageOps.autocontrast(Image.fromarray(gray), cutoff=1))
    threshold = max(10, int(otsu_threshold(gray) * 0.5))
    foreground = gray > threshold
    ys, xs = np.where(foreground)
    if len(xs) < 5 or len(ys) < 5:
        raise ValueError(
            "No se detectó un dígito con contraste suficiente. Use una imagen nítida."
        )

    cropped = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, min(20, int(round(w * scale))))
    new_h = max(1, min(20, int(round(h * scale))))
    resized = Image.fromarray(cropped).resize(
        (new_w, new_h), Image.Resampling.LANCZOS
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x0, y0 = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = np.asarray(resized, dtype=np.uint8)

    # Centrado aproximado por centro de masa, como preprocesamiento geométrico.
    mass = float(canvas.sum())
    if mass <= 0:
        raise ValueError("La imagen no contiene trazos visibles.")
    yy, xx = np.indices(canvas.shape)
    cx = float((xx * canvas).sum() / mass)
    cy = float((yy * canvas).sum() / mass)
    canvas = shift_image(canvas, int(round(13.5 - cx)), int(round(13.5 - cy)))

    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0)[None, None, :, :]
    return tensor, Image.fromarray(canvas)


# -----------------------------------------------------------------------------
# 4. Datos y evaluación
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_development_dataset():
    """Carga únicamente las 60.000 imágenes oficiales de entrenamiento."""
    return datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True,
        transform=transforms.ToTensor(),
    )


def calculate_metrics(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    if count == 0:
        raise ValueError("El conjunto de datos está vacío.")
    return {"loss": loss_sum / count, "accuracy": correct / count}


def train_one_fold(
    dataset,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    epochs: int,
    batch_size: int,
    fold_number: int,
    progress_callback=None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Entrena una partición y selecciona su mejor época según validación."""
    set_seed(SEED + fold_number)
    train_subset = Subset(dataset, train_indices.tolist())
    valid_subset = Subset(dataset, valid_indices.tolist())
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DigitMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_metrics = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        train_metrics = calculate_metrics(model, train_eval_loader)
        valid_metrics = calculate_metrics(model, valid_loader)
        row = {
            "fold": fold_number,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
        }
        history.append(row)

        # Selección basada exclusivamente en validación, nunca en test.
        if (best_metrics is None
            or valid_metrics["accuracy"] > best_metrics["valid_accuracy"]
            or (valid_metrics["accuracy"] == best_metrics["valid_accuracy"]
                and valid_metrics["loss"] < best_metrics["valid_loss"])):
            best_metrics = {
                **row,
                "model_state": copy.deepcopy(model.state_dict()),
            }
            best_state = copy.deepcopy(model.state_dict())

        if progress_callback:
            progress_callback(row)

    if best_state is None or best_metrics is None:
        raise RuntimeError(f"No se pudo entrenar la partición {fold_number}.")
    best_metrics["model_state"] = best_state
    return best_metrics, history


def run_five_fold_cv(epochs: int, batch_size: int, progress_callback=None):
    """Ejecuta 5-fold CV sobre train oficial; no carga ni consulta el test oficial."""
    if TEST_PATH.exists():
        raise RuntimeError(
            "El test oficial ya fue consultado en este experimento. No se permite "
            "volver a entrenar/seleccionar modelos usando ese mismo test."
        )
    dataset = load_development_dataset()
    labels = dataset.targets.cpu().numpy()
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    best_overall = None
    fold_summaries = []
    all_history = []

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        best_fold, history = train_one_fold(
            dataset, train_idx, valid_idx, epochs, batch_size, fold,
            progress_callback=progress_callback,
        )
        all_history.extend(history)
        summary = {k: v for k, v in best_fold.items() if k != "model_state"}
        fold_summaries.append(summary)
        if (best_overall is None
            or best_fold["valid_accuracy"] > best_overall["valid_accuracy"]
            or (best_fold["valid_accuracy"] == best_overall["valid_accuracy"]
                and best_fold["valid_loss"] < best_overall["valid_loss"])):
            best_overall = best_fold

    if best_overall is None:
        raise RuntimeError("La validación cruzada no produjo un modelo.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_overall["model_state"],
        "architecture": "784-64-1024-10",
        "selected_fold": int(best_overall["fold"]),
        "selected_epoch": int(best_overall["epoch"]),
        "validation_accuracy": float(best_overall["valid_accuracy"]),
        "validation_loss": float(best_overall["valid_loss"]),
        "input_shape": [1, 1, 28, 28],
        "pixel_domain": "MNIST: fondo negro=0, trazo claro=1, escala [0,1]",
    }
    torch.save(checkpoint, MODEL_PATH)

    import csv
    with CV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_history[0].keys()))
        writer.writeheader()
        writer.writerows(all_history)

    return fold_summaries, all_history, best_overall


def load_selected_model() -> DigitMLP:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Primero debe ejecutar la validación cruzada de 5 particiones.")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = DigitMLP().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_official_test_once() -> dict[str, float]:
    """Evalúa test oficial una sola vez y guarda el resultado para no reutilizarlo."""
    if TEST_PATH.exists():
        raise RuntimeError(
            "El conjunto test ya fue evaluado. Se conserva el resultado existente "
            "para evitar reutilizarlo durante ajustes del modelo."
        )
    test_dataset = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True,
        transform=transforms.ToTensor(),
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    model = load_selected_model()
    metrics = calculate_metrics(model, test_loader)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def predict_uploaded_image(image: Image.Image):
    model = load_selected_model()
    tensor, processed = preprocess_external_image(image)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor.to(DEVICE)), dim=1)[0].cpu().numpy()
    predicted = int(np.argmax(probabilities))
    return predicted, float(probabilities[predicted]), probabilities, processed


# -----------------------------------------------------------------------------
# 5. Interfaz Streamlit: entrenamiento, evaluación final e inferencia
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="MNIST | Teoría de Máquinas", page_icon="🔢", layout="wide")
    st.title("🔢 Reconocimiento de dígitos manuscritos – MNIST")
    st.caption("PyTorch · Validación cruzada 5-Fold · Streamlit · Localhost")
    st.write(
        "El sistema reconoce un dígito manuscrito del 0 al 9. El entrenamiento y "
        "el procesamiento se ejecutan localmente en este computador."
    )

    with st.sidebar:
        st.header("Configuración de entrenamiento")
        epochs = st.number_input("Épocas por partición", min_value=1, max_value=30, value=5, step=1)
        batch_size = st.selectbox("Tamaño del lote", [32, 64, 128, 256], index=2)
        st.caption(f"Dispositivo detectado: {DEVICE}")
        st.caption("5 particiones: 4 para entrenar y 1 para validar en cada ronda.")

    tab_train, tab_test, tab_predict, tab_info = st.tabs(
        ["1. Entrenamiento y validación", "2. Test final", "3. Reconocer imagen", "4. Información"]
    )

    with tab_train:
        st.subheader("Validación cruzada estratificada de cinco particiones")
        st.warning(
            "Las 10.000 imágenes oficiales de test no se cargan ni se consultan durante "
            "esta etapa. Cada época se compara con el conjunto de validación de su ronda."
        )
        if st.button(
            "Iniciar entrenamiento 5-Fold", type="primary",
            use_container_width=True, disabled=TEST_PATH.exists()
        ):
            if TEST_PATH.exists():
                st.error(
                    "El test oficial ya fue consultado. No se permite iniciar otro "
                    "entrenamiento con el mismo test, porque introduciría fuga de información."
                )
                st.stop()
            progress = st.progress(0.0)
            status = st.empty()
            total_steps = N_SPLITS * int(epochs)
            completed = 0
            fold_rows = []

            def update_progress(row):
                nonlocal completed
                completed += 1
                progress.progress(min(completed / total_steps, 1.0))
                status.write(
                    f"Partición {row['fold']}/5 · Época {row['epoch']}/{int(epochs)} · "
                    f"Train {row['train_accuracy']:.2%} · Valid {row['valid_accuracy']:.2%}"
                )

            try:
                with st.spinner("Descargando MNIST si es necesario y entrenando las cinco particiones..."):
                    folds, history, best = run_five_fold_cv(
                        int(epochs), int(batch_size), progress_callback=update_progress
                    )
                st.success("Entrenamiento y validación cruzada completados.")
                st.dataframe(folds, use_container_width=True)
                st.metric("Mejor exactitud de validación", f"{best['valid_accuracy']:.2%}")
                st.write(
                    f"Modelo seleccionado: partición {best['fold']}, época {best['epoch']}. "
                    "La selección se realizó exclusivamente con métricas de validación."
                )
                st.line_chart(
                    {"Train": [r["train_accuracy"] for r in history],
                     "Validación": [r["valid_accuracy"] for r in history]},
                    x_label="Épocas acumuladas de las cinco rondas", y_label="Exactitud"
                )
                st.info(f"Pesos guardados en: {MODEL_PATH.relative_to(BASE_DIR)}")
            except Exception as exc:
                st.error(f"No se pudo completar el entrenamiento: {exc}")

    with tab_test:
        st.subheader("Evaluación final con las 10.000 imágenes oficiales de test")
        st.write(
            "Esta evaluación se realiza después de seleccionar el modelo con validación. "
            "No se debe usar el resultado de test para modificar la arquitectura o elegir épocas."
        )
        if TEST_PATH.exists():
            metrics = json.loads(TEST_PATH.read_text(encoding="utf-8"))
            st.success("El test final ya fue evaluado. Se muestra el resultado guardado.")
            st.metric("Exactitud en test", f"{metrics['accuracy']:.2%}")
            st.metric("Pérdida en test", f"{metrics['loss']:.4f}")
        elif not MODEL_PATH.exists():
            st.info("Primero complete el entrenamiento y la validación cruzada.")
        elif st.button("Evaluar test final (una sola vez)", type="primary"):
            try:
                with st.spinner("Evaluando el conjunto test por primera vez..."):
                    metrics = evaluate_official_test_once()
                st.success("Evaluación final completada y guardada.")
                st.metric("Exactitud en test", f"{metrics['accuracy']:.2%}")
                st.metric("Pérdida en test", f"{metrics['loss']:.4f}")
            except Exception as exc:
                st.error(f"No se pudo evaluar test: {exc}")

    with tab_predict:
        st.subheader("Prueba con una imagen externa")
        st.write(
            "Cargue una imagen PNG, JPG o JPEG con un solo dígito. Se admiten "
            "números de tinta negra sobre papel blanco: el preprocesamiento invierte "
            "la polaridad para aproximarla al dominio MNIST (fondo negro=0, trazo claro=1)."
        )
        if not MODEL_PATH.exists():
            st.info("Primero entrene el modelo en la pestaña 1.")
        uploaded = st.file_uploader("Arrastre aquí la imagen o selecciónela", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            try:
                image = Image.open(uploaded).copy()
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Imagen original", use_container_width=True)
                if st.button("Reconocer dígito", type="primary"):
                    with st.spinner("Procesando imagen y realizando predicción..."):
                        digit, confidence, probabilities, processed = predict_uploaded_image(image)
                    with col2:
                        st.image(processed.resize((280, 280), Image.Resampling.NEAREST),
                                 caption="Entrada preparada de 28 × 28", width=280)
                    st.success(f"Dígito predicho: {digit}")
                    st.metric("Confianza estimada", f"{confidence:.2%}")
                    st.bar_chart({str(i): float(p) for i, p in enumerate(probabilities)})
            except Exception as exc:
                st.error(f"No se pudo procesar la imagen: {exc}")

    with tab_info:
        st.subheader("Arquitectura y metodología")
        st.code("Entrada 784 → Dense 64 (ReLU) → Dense 1024 (ReLU) → Salida 10", language="text")
        st.markdown(
            "- **Datos de desarrollo:** 60.000 imágenes MNIST oficiales de entrenamiento.\n"
            "- **Validación:** 5 particiones estratificadas; cada ronda usa 80 % para entrenar y 20 % para validar.\n"
            "- **Test:** 10.000 imágenes oficiales reservadas para una evaluación final.\n"
            "- **Optimizador:** Adam. **Función de pérdida:** CrossEntropyLoss.\n"
            "- **Dominio de píxeles:** fondo negro = 0; trazo claro = 1; valores entre 0 y 1."
        )
        st.warning(
            "La confianza es una probabilidad estimada, no una garantía. El sistema "
            "reconoce un solo dígito y no segmenta automáticamente números de varias cifras."
        )
        st.caption("Las imágenes cargadas se usan para inferencia y no se agregan al entrenamiento.")


if __name__ == "__main__":
    main()
