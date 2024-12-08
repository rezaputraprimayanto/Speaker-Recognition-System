import os
import librosa
import numpy as np

# Ekstraksi MFCC dari audio mentah
def extract_mfcc(signal, sr, n_mfcc=13, required_length=1131):
    """
    Ekstraksi MFCC dari data audio dengan padding/cropping agar panjang konsisten.
    """
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    padded_mfccs = np.zeros((n_mfcc, required_length))
    if mfccs.shape[1] < required_length:
        padded_mfccs[:, :mfccs.shape[1]] = mfccs
    else:
        padded_mfccs[:, :] = mfccs[:, :required_length]
    return padded_mfccs.T  # Transpose agar shape (timesteps, features)

# Proses folder dataset dengan menyimpan CSV dan NPZ
def process_and_save_dataset(audio_folder_path, output_folder, n_mfcc=13, required_length=1131, save_npz=True):
    """
    Proses dataset audio untuk membuat dataset multitask dengan MFCC.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    X = []
    y_commands = []
    y_speakers = []
    metadata = []

    for speaker in sorted(os.listdir(audio_folder_path)):  # Loop per speaker
        speaker_path = os.path.join(audio_folder_path, speaker)

        if os.path.isdir(speaker_path):
            for label in sorted(os.listdir(speaker_path)):  # Loop per label (commands)
                label_path = os.path.join(speaker_path, label)
                label_output_folder = os.path.join(output_folder, f"{label}_{speaker}")

                if not os.path.exists(label_output_folder):
                    os.makedirs(label_output_folder)

                if os.path.isdir(label_path):
                    for file_name in os.listdir(label_path):
                        if file_name.endswith('.wav'):
                            file_path = os.path.join(label_path, file_name)
                            try:
                                # Load audio file
                                signal, sr = librosa.load(file_path, sr=None)

                                # Ekstraksi MFCC
                                mfcc = extract_mfcc(signal, sr, n_mfcc=n_mfcc, required_length=required_length)
                                X.append(mfcc)

                                # Simpan label perintah dan speaker
                                y_commands.append(label)  # Label perintah
                                y_speakers.append(speaker)  # Label speaker

                                # Metadata
                                metadata.append({"speaker": speaker, "command": label, "file": file_name})

                                # Simpan MFCC ke file CSV
                                csv_path = os.path.join(label_output_folder, f"{os.path.splitext(file_name)[0]}.csv")
                                np.savetxt(csv_path, mfcc, delimiter=",")
                                print(f"[INFO] MFCC saved to: {csv_path}")

                            except Exception as e:
                                print(f"[ERROR] Gagal memproses file: {file_path}. Error: {e}")

    # Simpan dataset sebagai file .npz jika diinginkan
    if save_npz:
        npz_path = os.path.join(output_folder, "mfcc_dataset_multitask2.npz")
        np.savez(npz_path, X=np.array(X), y_commands=np.array(y_commands), y_speakers=np.array(y_speakers))
        print(f"[INFO] Dataset saved to {npz_path}")

        # Simpan metadata
        metadata_path = os.path.join(output_folder, "metadata.json")
        import json
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        print(f"[INFO] Metadata saved to {metadata_path}")

    return np.array(X), np.array(y_commands), np.array(y_speakers)

# Path folder audio dan output
audio_folder_path = r"C:\Users\aufam\Downloads\bisaaaa\downsampled_dataset"  # Folder audio Anda
output_folder = r"Multispeaker2"  # Output untuk file CSV dan NPZ

# Proses dan simpan dataset
X, y_commands, y_speakers = process_and_save_dataset(audio_folder_path, output_folder)

# Debugging dimensi dataset
print("Shape X:", X.shape)  # Harus (samples, timesteps, features)
print("Shape y_commands:", y_commands.shape)  # Harus (samples,)
print("Shape y_speakers:", y_speakers.shape)  # Harus (samples,)
print("Sample command labels:", np.unique(y_commands))
print("Sample speaker labels:", np.unique(y_speakers))
