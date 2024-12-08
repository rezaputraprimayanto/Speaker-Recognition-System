import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import joblib
from datetime import datetime

# Load model multitask dan encoder
model = tf.keras.models.load_model("multitask_speaker2_model.h5")
command_classes = joblib.load("command_encoder2.pkl")
speaker_classes = joblib.load("speaker_encoder2.pkl")

# Parameter untuk perekaman audio
SAMPLE_RATE = 16000
DURATION = 3  # Durasi rekaman dalam detik

# Fungsi untuk ekstraksi MFCC
def extract_mfcc(audio, sr, n_mfcc=13, required_length=1131):
    """
    Ekstraksi MFCC dari audio dengan padding/cropping agar panjangnya konsisten.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_padded = np.zeros((n_mfcc, required_length))
    if mfcc.shape[1] < required_length:
        mfcc_padded[:, :mfcc.shape[1]] = mfcc
    else:
        mfcc_padded[:, :] = mfcc[:, :required_length]
    return mfcc_padded.T

# Fungsi untuk melakukan prediksi perintah dan speaker
def predict(audio, sr):
    """
    Proses prediksi multitask: perintah dan speaker.
    """
    # Pastikan audio memiliki panjang tetap
    audio = librosa.util.fix_length(audio, size=sr * DURATION)

    # Ekstraksi fitur MFCC
    mfcc_features = extract_mfcc(audio, sr)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Tambahkan dimensi batch

    # Prediksi multitask
    predictions = model.predict(mfcc_features)
    command_pred = np.argmax(predictions[0])  # Output perintah
    confidence_command = np.max(predictions[0])
    speaker_pred = np.argmax(predictions[1])  # Output speaker
    confidence_speaker = np.max(predictions[1])

    # Validasi confidence
    if confidence_speaker < 0.7:
        speaker_label = "pengguna tidak dikenal"
    else:
        speaker_label = speaker_classes.inverse_transform([speaker_pred])[0]

    command_label = command_classes.inverse_transform([command_pred])[0]
    return command_label, confidence_command, speaker_label, confidence_speaker

# Runtime untuk deteksi suara
def run_realtime():
    print("[INFO] Sistem siap untuk mendeteksi perintah suara.")
    while True:
        try:
            # Rekam audio
            print("\n[INFO] Silakan berbicara...")
            audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()  # Tunggu hingga rekaman selesai
            audio = audio.flatten()  # Mengubah audio menjadi array 1D

            # Prediksi perintah dan speaker
            try:
                command, confidence_command, speaker, confidence_speaker = predict(audio, SAMPLE_RATE)
                print(f"[INFO] Perintah: {command} (Confidence: {confidence_command * 100:.2f}%), Speaker: {speaker} (Confidence: {confidence_speaker * 100:.2f}%)")

                # Validasi perintah dengan speaker
                if speaker == "pengguna tidak dikenal":
                    print("[WARNING] Pengguna tidak dikenal. Perintah ditolak.")
                elif command in ["buka_pintu", "kunci_pintu"]:
                    print(f"[INFO] Perintah '{command}' diterima dari speaker '{speaker}'.")
                else:
                    print(f"[INFO] Perintah '{command}' diterima.")
            except Exception as e:
                print(f"[ERROR] Tidak dapat memproses audio: {e}")

            # Jeda sebelum menerima perintah berikutnya
            print("[INFO] Sistem akan siap menerima perintah berikutnya dalam 3 detik...")
            sd.sleep(3000)
        except KeyboardInterrupt:
            print("\n[INFO] Sistem dihentikan.")
            break
        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan: {e}")
            continue

# Jalankan runtime
if __name__ == "__main__":
    run_realtime()