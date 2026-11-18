import json
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
notebook_path = os.path.join(root, "Edge_Assignment#4.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Deliverable Completion Cell\n",
        "\n",
        "Run this cell after the recorded training cells to save all model checkpoints, export all student and NAS artifacts, "
        "and generate benchmark CSV/JSON files."
    ],
}

code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Save all independently trained PyTorch checkpoints.\n",
        "checkpoint_map = {\n",
        "    'resnet50_teacher': teacher,\n",
        "    'mobilenet_v2_baseline': mobilenet_baseline,\n",
        "    'mobilenet_v2_distilled': mobilenet_distilled,\n",
        "    'squeezenet_baseline': squeezenet_baseline,\n",
        "    'squeezenet_distilled': squeezenet_distilled,\n",
        "    'efficientnet_lite0_baseline': efficientnet_baseline,\n",
        "    'efficientnet_lite0_distilled': efficientnet_distilled,\n",
        "    'nas_champion': nas_champion_model,\n",
        "}\n",
        "import os\n",
        "os.makedirs('checkpoints', exist_ok=True)\n",
        "for name, model in checkpoint_map.items():\n",
        "    torch.save(model.state_dict(), f'checkpoints/{name}.pth')\n",
        "    print(f'Saved checkpoints/{name}.pth')\n",
        "\n",
        "# Export every student and the NAS champion to ONNX.\n",
        "export_map = {\n",
        "    'mobilenet_v2_baseline': mobilenet_baseline,\n",
        "    'mobilenet_v2_distilled': mobilenet_distilled,\n",
        "    'squeezenet_baseline': squeezenet_baseline,\n",
        "    'squeezenet_distilled': squeezenet_distilled,\n",
        "    'efficientnet_lite0_baseline': efficientnet_baseline,\n",
        "    'efficientnet_lite0_distilled': efficientnet_distilled,\n",
        "    'nas_champion': nas_champion_model,\n",
        "}\n",
        "os.makedirs('onnx', exist_ok=True)\n",
        "for name, model in export_map.items():\n",
        "    export_student_to_onnx(model, f'onnx/{name}.onnx')\n",
        "\n",
        "# Save NAS history and the champion recipe.\n",
        "import csv, json\n",
        "with open('nas_results.csv', 'w', newline='') as f:\n",
        "    writer = csv.DictWriter(f, fieldnames=['config', 'params', 'accuracy', 'score'])\n",
        "    writer.writeheader()\n",
        "    writer.writerows(nas_results)\n",
        "with open('nas_champion.json', 'w') as f:\n",
        "    json.dump({'config': best_config, 'score': best_score, 'validation_accuracy': final_nas_acc}, f, indent=2)\n",
        "\n",
        "# CPU benchmark with ONNX Runtime, storage size, and peak-process-memory support.\n",
        "import time\n",
        "import numpy as np\n",
        "import psutil\n",
        "onnx_benchmark_rows = []\n",
        "dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)\n",
        "for name in export_map:\n",
        "    path = f'onnx/{name}.onnx'\n",
        "    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])\n",
        "    input_name = session.get_inputs()[0].name\n",
        "    for _ in range(10):\n",
        "        session.run(None, {input_name: dummy})\n",
        "    timings = []\n",
        "    process = psutil.Process()\n",
        "    rss_before = process.memory_info().rss\n",
        "    for _ in range(100):\n",
        "        start = time.perf_counter()\n",
        "        session.run(None, {input_name: dummy})\n",
        "        timings.append((time.perf_counter() - start) * 1000)\n",
        "    rss_after = process.memory_info().rss\n",
        "    size_bytes = os.path.getsize(path)\n",
        "    mean_ms = sum(timings) / len(timings)\n",
        "    onnx_benchmark_rows.append({\n",
        "        'model': name,\n",
        "        'provider': 'CPUExecutionProvider',\n",
        "        'batch_size': 1,\n",
        "        'mean_latency_ms': round(mean_ms, 3),\n",
        "        'throughput_images_per_sec': round(1000 / mean_ms, 2),\n",
        "        'model_size_bytes': size_bytes,\n",
        "        'peak_memory_delta_bytes': max(0, rss_after - rss_before),\n",
        "    })\n",
        "\n",
        "with open('benchmark_results.csv', 'w', newline='') as f:\n",
        "    writer = csv.DictWriter(f, fieldnames=onnx_benchmark_rows[0].keys())\n",
        "    writer.writeheader()\n",
        "    writer.writerows(onnx_benchmark_rows)\n",
        "\n",
        "print('Saved checkpoints, ONNX exports, NAS history, and CPU benchmark CSV.')"
    ],
}

notebook["cells"].extend([markdown, code])
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False)
