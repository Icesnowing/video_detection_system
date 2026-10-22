#!/bin/bash
# =============================================================================
# C++ Dependency Setup Script
# Downloads pre-built ONNX Runtime, OpenCV, and FFmpeg development files.
#
# For environments without system package manager access (no sudo),
# this fetches headers and libraries into third_party/ for CMake to find.
#
# After running this, build with:
#   cd cpp_version && mkdir -p build && cd build
#   cmake -DCMAKE_PREFIX_PATH="../../third_party/opencv" ..
#   make -j$(nproc)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY="$PROJECT_ROOT/third_party"
DOWNLOAD_DIR="$THIRD_PARTY/downloads"
mkdir -p "$THIRD_PARTY" "$DOWNLOAD_DIR"

ONNXRT_VERSION="1.20.1"
OPENCV_VERSION="4.10.0"

echo "============================================"
echo "  C++ Dependency Setup"
echo "  Project: $PROJECT_ROOT"
echo "============================================"
echo ""

# Utility: download with retries
download() {
    local url="$1"
    local output="$2"
    local name="$3"

    if [ -f "$output" ]; then
        echo "[INFO] $name already downloaded."
        return 0
    fi

    echo "[INFO] Downloading $name from $url ..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$output" "$url" 2>&1 || {
            echo "[WARN] wget failed: $url"
            return 1
        }
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$output" "$url" 2>&1 || {
            echo "[WARN] curl failed: $url"
            return 1
        }
    else
        echo "[ERROR] Neither wget nor curl found."
        return 1
    fi
    echo "[INFO] Downloaded: $name"
    return 0
}

# =============================================================================
# ONNX Runtime C++ SDK
# =============================================================================
setup_onnxruntime() {
    local ORT_DIR="$THIRD_PARTY/onnxruntime"

    if [ -f "$ORT_DIR/include/onnxruntime_cxx_api.h" ] && \
       [ -f "$ORT_DIR/lib/libonnxruntime.so" ]; then
        echo "[INFO] ONNX Runtime C++ SDK already installed."
        return 0
    fi

    echo "--- ONNX Runtime C++ SDK v$ONNXRT_VERSION ---"

    local ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRT_VERSION}/onnxruntime-linux-x64-${ONNXRT_VERSION}.tgz"
    local ORT_TGZ="$DOWNLOAD_DIR/onnxruntime-linux-x64-${ONNXRT_VERSION}.tgz"

    if ! download "$ORT_URL" "$ORT_TGZ" "ONNX Runtime SDK"; then
        echo "[WARN] Failed to download ONNX Runtime SDK."
        echo "  Manual download: https://github.com/microsoft/onnxruntime/releases"
        echo "  Extract to: $ORT_DIR"
        return 1
    fi

    echo "[INFO] Extracting ONNX Runtime..."
    rm -rf "$ORT_DIR"
    mkdir -p "$ORT_DIR"
    tar xzf "$ORT_TGZ" -C "$ORT_DIR" --strip-components=1

    # Verify
    if [ -f "$ORT_DIR/include/onnxruntime_cxx_api.h" ]; then
        echo "[INFO] ONNX Runtime SDK installed successfully."
        echo "  Headers:  $ORT_DIR/include/"
        echo "  Library:  $ORT_DIR/lib/libonnxruntime.so"
        echo ""
        echo "  >> Add to ~/.bashrc:"
        echo "  export LD_LIBRARY_PATH=\"$ORT_DIR/lib:\$LD_LIBRARY_PATH\""
    else
        echo "[ERROR] ONNX Runtime extraction failed."
        return 1
    fi
}

# =============================================================================
# OpenCV C++ (pre-built headers + libs)
# =============================================================================
setup_opencv() {
    local OCV_DIR="$THIRD_PARTY/opencv"

    if [ -d "$OCV_DIR/include/opencv4/opencv2" ] && \
       [ -f "$OCV_DIR/lib/libopencv_core.so" ]; then
        echo "[INFO] OpenCV C++ SDK already installed."
        return 0
    fi

    # Check system packages first
    if [ -d "/usr/include/opencv4/opencv2" ]; then
        echo "[INFO] System OpenCV headers found in /usr/include/opencv4/"
        mkdir -p "$OCV_DIR/include"
        ln -sf /usr/include/opencv4 "$OCV_DIR/include/opencv4" 2>/dev/null || true
    fi

    # Try extracting headers from Python OpenCV wheel
    # The opencv-python wheel on PyPI includes headers in the data directory
    local PY_CV2_DIR=$(python3 -c "import cv2, os; print(os.path.dirname(cv2.__file__))" 2>/dev/null)
    local PY_CV2_DATA="$PY_CV2_DIR/../../../opencv_data"

    if [ -d "$PY_CV2_DATA/include" ]; then
        echo "[INFO] Found OpenCV headers in Python package."
        mkdir -p "$OCV_DIR/include"
        cp -r "$PY_CV2_DATA/include/opencv4" "$OCV_DIR/include/" 2>/dev/null || {
            echo "[WARN] Could not copy OpenCV headers from Python package."
        }
    fi

    if [ -d "$OCV_DIR/include/opencv4" ]; then
        echo "[INFO] OpenCV headers available."
        # Find .so files from Python package
        local CV2_SO="$PY_CV2_DIR/cv2.abi3.so"
        if [ -f "$CV2_SO" ]; then
            mkdir -p "$OCV_DIR/lib"
            # Create symlink to the main opencv library
            ln -sf "$CV2_SO" "$OCV_DIR/lib/libopencv_world.so" 2>/dev/null || true
            echo "[INFO] OpenCV library linked from Python package."
        fi
    else
        echo "[WARN] OpenCV C++ headers not found."
        echo "  Install system package: sudo apt-get install libopencv-dev"
        echo "  Or download pre-built: https://opencv.org/releases/"
        echo ""
        echo "  The Python version runs without C++ OpenCV headers."
        return 1
    fi
}

# =============================================================================
# FFmpeg Development (headers + libs)
# =============================================================================
setup_ffmpeg() {
    echo "--- FFmpeg Development ---"

    local HAS_HEADERS=0
    local HAS_LIBS=0

    # Check system paths
    for dir in /usr/include /usr/local/include; do
        if [ -f "$dir/libavformat/avformat.h" ]; then
            echo "[INFO] System FFmpeg headers: $dir"
            HAS_HEADERS=1
            break
        fi
    done

    for dir in /usr/lib/x86_64-linux-gnu /usr/local/lib; do
        if [ -f "$dir/libavformat.so" ]; then
            echo "[INFO] System FFmpeg libraries: $dir"
            HAS_LIBS=1
            break
        fi
    done

    if [ $HAS_HEADERS -eq 0 ] || [ $HAS_LIBS -eq 0 ]; then
        echo "[WARN] FFmpeg development files not found."
        echo "  Install: sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev"
        echo ""
        echo "  NOTE: The Python version works with just the ffmpeg binary"
        echo "  (already present in third_party/ffmpeg/ffmpeg)."
        echo ""
        echo "  The C++ version requires FFmpeg dev libraries for native API."
        echo "  Without them, use the Python version instead."
    else
        echo "[INFO] FFmpeg development files found."
    fi
}

# =============================================================================
# Main
# =============================================================================
echo ""
setup_onnxruntime
echo ""
setup_opencv
echo ""
setup_ffmpeg

echo ""
echo "============================================"
echo "  Dependency setup complete."
echo ""
if [ -f "$THIRD_PARTY/onnxruntime/include/onnxruntime_cxx_api.h" ] && \
   [ -d "$THIRD_PARTY/opencv/include/opencv4" ]; then
    echo "  Build C++ version:"
    echo "    cd cpp_version && mkdir -p build && cd build"
    echo "    cmake -DCMAKE_PREFIX_PATH=\"../../third_party/onnxruntime;../../third_party/opencv\" .."
    echo "    make -j\$(nproc)"
    echo ""
    echo "  Then run:"
    echo "    ./video_detection --source video --path ../video_output/test_video.mp4"
else
    echo "  Some C++ dependencies are missing."
    echo "  The Python version is fully functional:"
    echo "    cd python_version && python3 run.py --source video --path ../video_output/test_video.mp4"
fi
echo "============================================"
