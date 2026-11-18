import csv
import json
import os
import time
import tracemalloc

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ONNX_PATH = os.path.join(ROOT, "mobilenet_distilled.onnx")

kd_results = [
    ("ResNet-50 teacher", "CIFAR-100", "-", 52.47),
    ("MobileNetV2 baseline", "CIFAR-100", "Baseline", 40.92),
    ("MobileNetV2 distilled", "CIFAR-100", "Distilled", 46.26),
    ("SqueezeNet baseline", "CIFAR-100", "Baseline", 16.09),
    ("SqueezeNet distilled", "CIFAR-100", "Distilled", 34.95),
    ("EfficientNet-Lite0 baseline", "CIFAR-100", "Baseline", 46.37),
    ("EfficientNet-Lite0 distilled", "CIFAR-100", "Distilled", 50.32),
]

nas_results = [
    (1, 3, [32, 32, 32], [3, 3, 3], ["depthwise_separable", "standard", "depthwise_separable"], "leaky_relu", 14143, 2.18, 90.2953),
    (2, 4, [64, 16, 128, 32], [5, 3, 3, 3], ["standard", "standard", "standard", "standard"], "silu", 73092, 4.44, 53.4347),
    (3, 5, [16, 32, 16, 128, 64], [5, 5, 3, 5, 5], ["depthwise_separable", "depthwise_separable", "depthwise_separable", "depthwise_separable", "standard"], "relu", 216095, 3.97, 17.5590),
    (4, 4, [16, 64, 32, 16], [3, 3, 5, 3], ["depthwise_separable", "standard", "depthwise_separable", "standard"], "leaky_relu", 19503, 3.64, 123.3773),
    (5, 5, [16, 128, 16, 64, 128], [5, 3, 3, 3, 5], ["depthwise_separable", "standard", "standard", "depthwise_separable", "depthwise_separable"], "silu", 61551, 5.19, 72.5357),
]


def write_csvs():
    with open(os.path.join(ROOT, "kd_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "regime", "validation_accuracy_percent"])
        w.writerows(kd_results)

    with open(os.path.join(ROOT, "nas_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "depth", "channels", "kernels", "conv_types", "activation", "parameters", "validation_accuracy_percent", "score"])
        w.writerows(nas_results)


def benchmark_onnx():
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return None

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    data = np.random.randn(1, 3, 32, 32).astype(np.float32)
    input_name = session.get_inputs()[0].name

    for _ in range(10):
        session.run(None, {input_name: data})

    timings = []
    tracemalloc.start()
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {input_name: data})
        timings.append((time.perf_counter() - start) * 1000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    size_bytes = os.path.getsize(ONNX_PATH)
    external_data = ONNX_PATH + ".data"
    if os.path.exists(external_data):
        size_bytes += os.path.getsize(external_data)

    mean_ms = sum(timings) / len(timings)
    p50 = sorted(timings)[len(timings) // 2]
    p95 = sorted(timings)[int(len(timings) * 0.95) - 1]
    return {
        "provider": "CPUExecutionProvider",
        "batch_size": 1,
        "mean_latency_ms": round(mean_ms, 3),
        "p50_latency_ms": round(p50, 3),
        "p95_latency_ms": round(p95, 3),
        "throughput_images_per_sec": round(1000 / mean_ms, 2),
        "model_size_bytes": size_bytes,
        "model_size_mb": round(size_bytes / (1024 * 1024), 3),
        "python_traced_peak_bytes": peak,
        "warmup_runs": 10,
        "measured_runs": 100,
    }


def make_chart():
    labels = [r[0].replace(" distilled", "\ndistilled").replace(" baseline", "\nbaseline") for r in kd_results]
    values = [r[3] for r in kd_results]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=8)
    plt.ylabel("Validation accuracy (%)")
    plt.ylim(0, 60)
    plt.title("CIFAR-100 Results Recorded in the Notebook")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    chart = os.path.join(ROOT, "accuracy_chart.png")
    plt.savefig(chart, dpi=180)
    plt.close()
    return chart


def build_report(benchmark):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        os.path.join(ROOT, "Technical_Report.pdf"),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Edge Computing and Intelligent Systems - Assignment 4",
    )
    story = []
    story.append(Paragraph("Edge Computing & Intelligent Systems", styles["Title"]))
    story.append(Paragraph("Assignment 4 - Knowledge Distillation and Neural Architecture Search", styles["Heading1"]))
    story.append(Paragraph("Experimental report based on the executed notebook outputs", styles["Heading2"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Experimental Environment", styles["Heading2"]))
    story.append(Paragraph(
        "The experiments were executed on Kaggle using one NVIDIA T4 GPU. The notebook used PyTorch, "
        "torchvision, CIFAR-100, Adam optimization, cosine annealing, and a 128-sample training batch.",
        styles["BodyText"],
    ))

    story.append(Paragraph("2. Knowledge Distillation", styles["Heading2"]))
    story.append(Paragraph(
        "The ResNet-50 teacher was adapted to 32 x 32 CIFAR-100 inputs and trained for five epochs. "
        "Its final validation accuracy was 52.47%. MobileNetV2, SqueezeNet, and EfficientNet-B0 (used as the "
        "EfficientNet-Lite0 backbone) were trained for 10 epochs in baseline and distilled regimes.",
        styles["BodyText"],
    ))
    table_data = [["Model", "Training", "Val. accuracy (%)"]]
    table_data += [[name, regime, f"{accuracy:.2f}"] for name, _, regime, accuracy in kd_results]
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF2F8")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("3. Distillation Improvements", styles["Heading2"]))
    story.append(Paragraph(
        "Compared with baseline training, MobileNetV2 improved by 5.34 percentage points, "
        "SqueezeNet improved by 18.86 percentage points, and EfficientNet-Lite0 improved by 3.95 percentage points.",
        styles["BodyText"],
    ))
    story.append(Paragraph(
        "The strongest absolute result was distilled EfficientNet-Lite0 at 50.32%. Distilled MobileNetV2 reached "
        "46.26%, while distilled SqueezeNet reached 34.95%. The loss implementation combines cross-entropy, "
        "temperature-scaled KL divergence at T=4, alpha=0.3, beta=0.7, and an optional feature-MSE term.",
        styles["BodyText"],
    ))
    chart = make_chart()
    from reportlab.platypus import Image
    story.append(Image(chart, width=17 * cm, height=8.5 * cm))

    story.append(Paragraph("4. NAS Results", styles["Heading2"]))
    story.append(Paragraph(
        "The notebook sampled five random searchable CNN configurations and used a one-epoch, 5,000-image proxy "
        "training run. Trial 4 produced the best recorded multi-objective score (123.3773), with 19,503 parameters "
        "and 3.64% proxy validation accuracy. Its final fully trained CIFAR-100 validation accuracy was 20.71%. "
        "The champion configuration was depth 4, channels [16, 64, 32, 16], kernels [3, 3, 5, 3], convolution types "
        "[depthwise separable, standard, depthwise separable, standard], and LeakyReLU activation.",
        styles["BodyText"],
    ))

    nas_table = [["Trial", "Params", "Proxy acc.", "Score"]]
    nas_table += [[str(r[0]), f"{r[6]:,}", f"{r[7]:.2f}", f"{r[8]:.4f}"] for r in nas_results]
    table = Table(nas_table, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF2F8")]),
    ]))
    story.append(table)

    story.append(Paragraph("5. ONNX CPU Benchmark", styles["Heading2"]))
    if benchmark:
        benchmark_table = Table([
            ["Metric", "Measured value"],
            ["Provider", benchmark["provider"]],
            ["Batch size", str(benchmark["batch_size"])],
            ["Mean latency", f"{benchmark['mean_latency_ms']:.3f} ms"],
            ["Median latency", f"{benchmark['p50_latency_ms']:.3f} ms"],
            ["P95 latency", f"{benchmark['p95_latency_ms']:.3f} ms"],
            ["Throughput", f"{benchmark['throughput_images_per_sec']:.2f} images/s"],
            ["Combined ONNX size", f"{benchmark['model_size_mb']:.3f} MB"],
        ])
    else:
        benchmark_table = Table([
            ["Metric", "Value"],
            ["Provider", "CPUExecutionProvider"],
            ["Status", "Benchmark dependency unavailable; rerun generate_report.py in the runtime with onnxruntime installed"],
        ])
    benchmark_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF2F8")]),
    ]))
    story.append(benchmark_table)

    story.append(Paragraph("6. Deliverables and Limitations", styles["Heading2"]))
    story.append(Paragraph(
        "Delivered artifacts include the notebook, exported distilled MobileNetV2 ONNX model and external weights, "
        "accuracy CSVs, NAS history CSV, this technical report, and README. The notebook should be rerun once "
        "to generate/save every baseline and distilled PyTorch checkpoint, the remaining ONNX exports, NAS search "
        "CSV/JSON, and a hardware benchmark CSV. The PDF's strict requirements call for CIFAR-10 NAS, evolutionary "
        "or Optuna search, actual CPU latency/storage rejection, and peak-memory measurement. The recorded executed "
        "NAS search used CIFAR-100 and random sampling rather than that stricter configuration.",
        styles["BodyText"],
    ))

    doc.build(story)


def main():
    write_csvs()
    benchmark = benchmark_onnx()
    if benchmark:
        with open(os.path.join(ROOT, "benchmark_results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(benchmark.keys()))
            w.writeheader()
            w.writerow(benchmark)

    summary = {
        "hardware_training": "Kaggle 1x NVIDIA T4 GPU",
        "dataset_recorded": "CIFAR-100",
        "teacher_final_validation_accuracy_percent": 52.47,
        "knowledge_distillation": kd_results,
        "nas": nas_results,
        "nas_champion": {
            "trial": 4,
            "parameters": 19503,
            "proxy_accuracy_percent": 3.64,
            "score": 123.3773,
            "final_validation_accuracy_percent": 20.71,
            "config": {
                "depth": 4,
                "channels": [16, 64, 32, 16],
                "kernels": [3, 3, 5, 3],
                "conv_types": ["depthwise_separable", "standard", "depthwise_separable", "standard"],
                "activation": "leaky_relu",
            },
        },
        "onnx_cpu_benchmark": benchmark,
    }
    with open(os.path.join(ROOT, "report_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    build_report(benchmark)


if __name__ == "__main__":
    main()
