# Edge Assignment #4 Deliverables

## Environment

- Training: Kaggle 1x NVIDIA T4 GPU
- Framework: PyTorch and torchvision
- Export/runtime: ONNX Opset 18 and ONNX Runtime CPU provider
- Dataset used by the recorded notebook runs: CIFAR-100

## Files

- `Edge_Assignment#4.ipynb`: source implementation and recorded training outputs
- `mobilenet_distilled.onnx`: exported distilled MobileNetV2 model
- `mobilenet_distilled.onnx.data`: external ONNX model weights
- `technical_report.pdf`: implementation and experimental report
- `kd_results.csv`: final knowledge-distillation accuracy comparison
- `nas_results.csv`: NAS trial history recorded in the notebook
- `benchmark_results.csv`: CPU benchmark for the exported ONNX artifact
- `report_summary.json`: machine-readable experiment summary
- `scripts/generate_report.py`: regenerates the report and benchmark CSV

## Reproducible execution

1. Open `Edge_Assignment#4.ipynb` in Kaggle or Jupyter.
2. Enable GPU acceleration and install the notebook dependencies.
3. Run all cells in order.
4. Run `python scripts/generate_report.py` locally to refresh benchmark/report artifacts.

## Assignment-status note

The recorded notebook completes Question 1 and records all required MobileNetV2, SqueezeNet, and EfficientNet-Lite0 training runs. Feature-loss support is implemented in `DistillationLoss`, although the recorded training calls currently pass no intermediate feature tensors. The exported artifact is the distilled MobileNetV2 model.

The original NAS run uses CIFAR-100, random search, and a proxy accuracy/parameter score. A stricter execution of the PDF should use CIFAR-10, a stateful evolutionary or Optuna search, actual CPU latency/storage constraints, and save/export the champion. The notebook's appended completion cell provides this remaining implementation path.
