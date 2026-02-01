"""Simplified AI intrusion detection agent for an academic demo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - handled for user guidance
    tf = None


@dataclass
class PredictionResult:
    label: str
    confidence: float


class IntrusionDetectionAgent:
    """Loads an LSTM model and provides predictions for traffic features."""

    def __init__(self, model_path: str | Path, feature_columns: Iterable[str]):
        self.model_path = Path(model_path)
        self.feature_columns = list(feature_columns)
        self.model = self._load_or_create_model()

    def _build_default_model(self) -> "tf.keras.Model":
        """Build a small LSTM model used for demo purposes."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(1, len(self.feature_columns))),
                tf.keras.layers.LSTM(16, activation="tanh"),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def _load_or_create_model(self) -> "tf.keras.Model":
        """Load a saved LSTM model, or create/save one for first-time setup."""
        if tf is None:
            raise ImportError(
                "TensorFlow is required to load the LSTM model. "
                "Install it with `pip install tensorflow` before running the dashboard."
            )

        if self.model_path.exists():
            try:
                return tf.keras.models.load_model(self.model_path)
            except Exception:
                # If the saved file is missing/corrupted, rebuild a fresh demo model.
                self.model_path.unlink(missing_ok=True)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model = self._build_default_model()
        model.save(self.model_path)
        return model

    def preprocess(self, features: Iterable[float]) -> np.ndarray:
        """Convert raw feature values to the model input shape."""
        values = np.array(list(features), dtype=np.float32)
        return values.reshape(1, 1, -1)

    def predict(self, features: Iterable[float]) -> PredictionResult:
        """Return a NORMAL/ATTACK label and confidence score."""
        inputs = self.preprocess(features)
        probability = float(self.model.predict(inputs, verbose=0)[0][0])
        label = "ATTACK" if probability >= 0.5 else "NORMAL"
        confidence = probability if label == "ATTACK" else 1.0 - probability
        return PredictionResult(label=label, confidence=confidence)


def load_agent(model_path: str | Path, feature_columns: Iterable[str]) -> IntrusionDetectionAgent:
    """Helper to build the agent (used by the Streamlit dashboard)."""
    return IntrusionDetectionAgent(model_path, feature_columns)


__all__ = ["IntrusionDetectionAgent", "PredictionResult", "load_agent"]
