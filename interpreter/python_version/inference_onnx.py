"""
ONNX Runtime inference engine (CPU-optimized replacement for TensorRT).

Loads a YOLOv11 ONNX model and runs inference with configurable providers.
On CPU: uses CPUExecutionProvider with thread pool optimizations.
"""

import numpy as np
import onnxruntime as ort
from typing import List, Optional, Dict, Any
import time


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
        graph_optimization_level: str = "99",
        enable_profiling: bool = False,
    ):
        """
        Initialize ONNX Runtime inference engine.

        Args:
            model_path:                Path to ONNX model file
            provider:                  Execution provider name
            intra_op_threads:          Number of threads for parallel nodes (0 = auto)
            inter_op_threads:          Number of threads for parallel exec (0 = auto)
            graph_optimization_level:  ORT_DISABLE_ALL=0, ORT_ENABLE_BASIC=1,
                                       ORT_ENABLE_EXTENDED=2, ORT_ENABLE_ALL=99
            enable_profiling:          Enable ORT profiling output
        """
        self.model_path = model_path

        # Configure session options
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, "ORT_ENABLE_ALL", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        self.sess_options.enable_profiling = enable_profiling

        if intra_op_threads > 0:
            self.sess_options.intra_op_num_threads = intra_op_threads
        if inter_op_threads > 0:
            self.sess_options.inter_op_num_threads = inter_op_threads

        # Configure provider-specific options
        provider_options = []
        if provider == "CPUExecutionProvider":
            opts = {}
            if intra_op_threads > 0:
                opts["intra_op_num_threads"] = str(intra_op_threads)
            if inter_op_threads > 0:
                opts["inter_op_num_threads"] = str(inter_op_threads)
            provider_options = [opts]
        elif provider == "CUDAExecutionProvider":
            provider_options = [{"device_id": "0"}]

        self.providers = [provider]

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.input_shape: tuple = ()
        self.output_shape: tuple = ()

        # Warmup/benchmark state
        self._warmup_done = False
        self._latency_samples: List[float] = []

    def load(self) -> bool:
        """Load the ONNX model and create an inference session."""
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=self.sess_options,
                providers=self.providers,
            )

            input_meta = self.session.get_inputs()[0]
            output_meta = self.session.get_outputs()[0]

            self.input_name = input_meta.name
            self.output_name = output_meta.name
            self.input_shape = input_meta.shape
            self.output_shape = output_meta.shape

            print(f"[Inference] Model loaded: {self.model_path}")
            print(f"  Input:  {self.input_name} {self.input_shape} {input_meta.type}")
            print(f"  Output: {self.output_name} {self.output_shape} {output_meta.type}")
            print(f"  Provider: {self.session.get_providers()}")

            return True
        except Exception as e:
            print(f"[Inference] ERROR loading model: {e}")
            return False

    def infer(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a single input tensor.

        Args:
            input_tensor:  (1, 3, H, W) float32 numpy array

        Returns:
            Dict mapping output names to numpy arrays
        """
        if self.session is None:
            raise RuntimeError("Session not loaded. Call load() first.")

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return {self.output_name: outputs[0]}

    def infer_timed(self, input_tensor: np.ndarray) -> tuple:
        """Infer with latency measurement."""
        t0 = time.perf_counter()
        result = self.infer(input_tensor)
        latency = (time.perf_counter() - t0) * 1000.0  # ms
        self._latency_samples.append(latency)
        return result, latency

    def warmup(self, num_runs: int = 10):
        """Warmup the inference session with dummy data."""
        if self.session is None:
            raise RuntimeError("Session not loaded. Call load() first.")

        # Determine static shape; fallback to (1, 3, 640, 640)
        shape = list(self.input_shape)
        for i, d in enumerate(shape):
            if isinstance(d, str) or d is None or d < 0:
                shape[i] = 1 if i == 0 else (3 if i == 1 else 640)

        dummy = np.random.randn(*shape).astype(np.float32)

        print(f"[Inference] Warming up ({num_runs} runs)...")
        for i in range(num_runs):
            self.infer(dummy)
        self._warmup_done = True
        print("[Inference] Warmup complete.")

    def benchmark(self, num_runs: int = 100):
        """Benchmark inference throughput and latency."""
        if self.session is None:
            raise RuntimeError("Session not loaded.")

        shape = list(self.input_shape)
        for i, d in enumerate(shape):
            if isinstance(d, str) or d is None or d < 0:
                shape[i] = 1 if i == 0 else (3 if i == 1 else 640)

        dummy = np.random.randn(*shape).astype(np.float32)

        latencies = []
        for _ in range(num_runs):
            _, lat = self.infer_timed(dummy)
            latencies.append(lat)

        latencies = np.array(latencies)
        print(f"[Inference] Benchmark ({num_runs} runs):")
        print(f"  Mean:   {latencies.mean():.3f} ms")
        print(f"  Median: {np.median(latencies):.3f} ms")
        print(f"  Min:    {latencies.min():.3f} ms")
        print(f"  Max:    {latencies.max():.3f} ms")
        print(f"  FPS:    {1000.0 / latencies.mean():.1f}")

        return latencies

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)

    def close(self):
        """Release resources (no-op for ORT but good practice)."""
        self.session = None
