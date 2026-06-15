from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def build_hog_descriptor(image_size: int) -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=(image_size, image_size),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def extract_features(image_path: Path, hog: cv2.HOGDescriptor, image_size: int) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    features = hog.compute(resized)
    if features is None:
        return None
    return features.reshape(-1).astype(np.float32)


def load_samples(dataset_dir: Path) -> Tuple[List[Sample], List[str]]:
    samples: List[Sample] = []
    labels: List[str] = []
    for label_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        label = label_dir.name.strip()
        if not label:
            continue
        labels.append(label)
        for image_path in sorted(label_dir.glob("*.jpg")):
            samples.append(Sample(path=image_path, label=label))
    return samples, labels


def batchify(items: Sequence[Sample], batch_size: int) -> Iterable[List[Sample]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def make_matrix(batch: Sequence[Sample], hog: cv2.HOGDescriptor, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for sample in batch:
        vector = extract_features(sample.path, hog, image_size)
        if vector is None:
            continue
        features.append(vector)
        labels.append(sample.label)
    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=object)
    return np.stack(features).astype(np.float32), np.asarray(labels)


def evaluate(classifier, items: Sequence[Sample], hog: cv2.HOGDescriptor, image_size: int, batch_size: int) -> float:
    y_true: List[str] = []
    y_pred: List[str] = []
    for batch in batchify(items, batch_size):
        X, y = make_matrix(batch, hog, image_size)
        if len(X) == 0:
            continue
        preds = classifier.predict(X)
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())
    if not y_true:
        return 0.0
    return float(accuracy_score(y_true, y_pred))


def train(
    dataset_dir: Path,
    output_model: Path,
    output_labels: Path,
    image_size: int,
    epochs: int,
    batch_size: int,
    test_size: float,
    random_state: int,
) -> dict:
    samples, labels = load_samples(dataset_dir)
    if not samples:
        raise RuntimeError(f"No JPG samples found under {dataset_dir}")

    label_counts = Counter(sample.label for sample in samples)
    print(f"Loaded {len(samples)} images across {len(labels)} labels.")
    print(f"Min/Max samples per label: {min(label_counts.values())}/{max(label_counts.values())}")

    train_items, val_items = train_test_split(
        samples,
        test_size=test_size,
        random_state=random_state,
        stratify=[sample.label for sample in samples],
    )

    hog = build_hog_descriptor(image_size)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(labels),
        y=[sample.label for sample in train_items],
    )
    class_weight_map = {label: float(weight) for label, weight in zip(labels, class_weights)}

    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        learning_rate="optimal",
        random_state=random_state,
        class_weight=class_weight_map,
        average=True,
    )

    classes = np.asarray(labels)
    best_classifier = None
    best_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(random_state + epoch)
        order = rng.permutation(len(train_items))
        shuffled = [train_items[index] for index in order]
        first_batch = True
        seen = 0

        for batch in batchify(shuffled, batch_size):
            X, y = make_matrix(batch, hog, image_size)
            if len(X) == 0:
                continue
            if first_batch:
                classifier.partial_fit(X, y, classes=classes)
                first_batch = False
            else:
                classifier.partial_fit(X, y)
            seen += len(X)

        val_accuracy = evaluate(classifier, val_items, hog, image_size, batch_size)
        print(f"Epoch {epoch}/{epochs} - trained on {seen} samples - val_accuracy={val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_classifier = copy.deepcopy(classifier)

    if best_classifier is None:
        best_classifier = classifier
        best_accuracy = evaluate(best_classifier, val_items, hog, image_size, batch_size)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "classifier": best_classifier,
        "labels": labels,
        "image_size": image_size,
        "validation_accuracy": best_accuracy,
        "sample_count": len(samples),
        "trained_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_dir": str(dataset_dir),
        "batch_size": batch_size,
        "epochs": epochs,
    }
    joblib.dump(payload, output_model)
    output_labels.write_text(json.dumps({"labels": labels}, indent=2), encoding="utf-8")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ISL classifier from the Indian dataset folder.")
    parser.add_argument("--dataset", default="Indian", help="Dataset root directory.")
    parser.add_argument("--output-model", default="runtime/isl_model.joblib", help="Where to save the trained model.")
    parser.add_argument("--output-labels", default="runtime/isl_labels.json", help="Where to save the label list.")
    parser.add_argument("--image-size", type=int, default=64, help="Resize size used by HOG.")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    payload = train(
        dataset_dir=dataset_dir,
        output_model=Path(args.output_model).resolve(),
        output_labels=Path(args.output_labels).resolve(),
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("Saved model to:", Path(args.output_model).resolve())
    print("Saved labels to:", Path(args.output_labels).resolve())
    print(f"Validation accuracy: {payload['validation_accuracy']:.4f}")


if __name__ == "__main__":
    main()
