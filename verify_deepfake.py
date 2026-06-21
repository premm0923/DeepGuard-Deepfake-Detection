import os
import cv2
import numpy as np
import tensorflow as tf

# =====================================================================
# 0. LOCAL INFRASTRUCTURE CONFIGURATIONS
# =====================================================================
# Everything is in the same folder, so relative paths work perfectly!
MODEL_PATH      = "./deepguard_final_engine_production.keras"
FRAMES          = 30
IMG_H, IMG_W    = 128, 128

# 🎬 Change this filename if you want to test a different video file later
TEST_VIDEO_PATH = "./a.mp4"

# =====================================================================
# 1. REGISTER CUSTOM KERAS METRIC DESERIALIZERS
# =====================================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class SafePrecision(tf.keras.metrics.Metric):
    def __init__(self, name="precision", **kwargs): super().__init__(name=name, **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None): pass
    def result(self): return 0.0

@tf.keras.utils.register_keras_serializable(package="Custom")
class SafeRecall(tf.keras.metrics.Metric):
    def __init__(self, name="recall", **kwargs): super().__init__(name=name, **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None): pass
    def result(self): return 0.0

# =====================================================================
# 2. LOCAL MULTI-DIMENSIONAL VIDEO FRAME DECODER
# =====================================================================
def load_and_preprocess_video(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Input target video file not found at: {path}\n💡 Make sure to place an 'a.mp4' video inside your project folder!")
        
    cap = cv2.VideoCapture(path)
    frames_list = []
    
    while len(frames_list) < FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_W, IMG_H))
        frames_list.append(frame.astype(np.float32) / 255.0)
    cap.release()
    
    if len(frames_list) == 0:
        raise ValueError("❌ OpenCV completely failed to unpack target video matrix.")
        
    while len(frames_list) < FRAMES:
        frames_list.append(frames_list[-1])
        
    return np.expand_dims(np.array(frames_list[:FRAMES], dtype=np.float32), axis=0)

# =====================================================================
# 3. COMPUTATIONAL GRAPH EVALUATION EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("📦 Initializing local deep learning dependencies...")
    custom_mapping = {"SafePrecision": SafePrecision, "SafeRecall": SafeRecall}
    
    # Load model instance cleanly without compile bindings
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_mapping, compile=False)
    
    print(f"🎬 Processing frames for target array: {os.path.basename(TEST_VIDEO_PATH)}")
    try:
        input_tensor = load_and_preprocess_video(TEST_VIDEO_PATH)
        
        print("🧠 Running forward pass prediction matrices...")
        prediction = model.predict(input_tensor, verbose=0)[0][0]
        
        print("\n" + "="*50)
        print("📊 LOCAL DEEPGUARD SECURITY ANALYSIS REPORT")
        print("="*50)
        print(f"Prediction Probability: {prediction:.6f}")
        if prediction > 0.35:
            print(f"❌ VERDICT: DEEPFAKE DETECTED (Confidence: {prediction * 100:.2f}%)")
        else:
            print(f"✅ VERDICT: REAL VIDEO VERIFIED (Confidence: {(1 - prediction) * 100:.2f}%)")
        print("="*50)
    except Exception as e:
        print(f"\n{str(e)}")
