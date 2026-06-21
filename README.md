# DeepGuard: Spatiotemporal Audio-Visual Deepfake Detection

DeepGuard is a custom computer vision framework designed to isolate and detect localized video alterations. By processing multimedia feeds sequentially, the engine evaluates micro-expressions and frame-to-frame pixel consistency to determine stream authenticity.

---

## ⚡ Core Architecture

The processing pipeline treats video files as multi-dimensional matrices, splitting features across two distinct neural components:

* **Spatial Processing (CNN):** A high-throughput Convolutional Neural Network backbone analyzes frame-level pixel arrays to catch local textures, lighting anomalies, and artificial blending boundaries.
* **Temporal Tracking (LSTM):** A sequential Long Short-Term Memory network maps the extracted spatial feature vectors over a timeline, identifying structural gaps or motion glitches across continuous playback.

---

## 🛠️ Local System Configuration

To clone the source blueprint and configure the required local python runtime environment, execute the following commands:

```powershell
# Clone the repository
git clone [https://github.com/premm0923/DeepGuard-Deepfake-Detection.git](https://github.com/premm0923/DeepGuard-Deepfake-Detection.git)

# Move into the workspace directory
cd DeepGuard-Deepfake-Detection

# Install the exact framework dependencies
pip install -r requirements.txt
