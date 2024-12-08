import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# Load dataset dari .npz
dataset_path = r"C:\Users\aufam\Downloads\bisaaaa\Multispeaker2\mfcc_dataset_multitask2.npz"
data = np.load(dataset_path)
X = data['X']  # Fitur MFCC
y_commands = data['y_commands']  # Label perintah
y_speakers = data['y_speakers']  # Label speaker

# Encode label untuk perintah dan speaker
command_encoder = LabelEncoder()
speaker_encoder = LabelEncoder()

y_commands_encoded = command_encoder.fit_transform(y_commands)
y_speakers_encoded = speaker_encoder.fit_transform(y_speakers)

# Split dataset menjadi training dan testing
X_train, X_test, y_commands_train, y_commands_test, y_speakers_train, y_speakers_test = train_test_split(
    X, y_commands_encoded, y_speakers_encoded, test_size=0.2, random_state=42)

# Debugging dimensi data
print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Number of command classes:", len(command_encoder.classes_))
print("Number of speaker classes:", len(speaker_encoder.classes_))

# Definisikan input untuk CNN
input_layer = Input(shape=(X.shape[1], X.shape[2]))

# CNN Layers
x = Conv1D(64, 3, activation='relu')(input_layer)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)
x = Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.01))(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)

# Dense layers sebelum output
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

# Output untuk prediksi perintah
command_output = Dense(len(command_encoder.classes_), activation='softmax', name='command_output')(x)

# Output untuk prediksi speaker
speaker_output = Dense(len(speaker_encoder.classes_), activation='softmax', name='speaker_output')(x)

# Model multitask
model = Model(inputs=input_layer, outputs=[command_output, speaker_output])

# Compile model dengan metrik per output
model.compile(optimizer='adam',
              loss={'command_output': 'sparse_categorical_crossentropy',
                    'speaker_output': 'sparse_categorical_crossentropy'},
              metrics={'command_output': ['accuracy'],
                       'speaker_output': ['accuracy']})

# Latih model
history = model.fit(
    X_train,
    {'command_output': y_commands_train, 'speaker_output': y_speakers_train},
    validation_split=0.2,
    epochs=40,
    batch_size=16,
)

# Evaluasi model pada data testing
results = model.evaluate(
    X_test,
    {'command_output': y_commands_test, 'speaker_output': y_speakers_test}
)
print(f"Test Loss (Command): {results[1]:.4f}, Test Accuracy (Command): {results[3] * 100:.2f}%")
print(f"Test Loss (Speaker): {results[2]:.4f}, Test Accuracy (Speaker): {results[4] * 100:.2f}%")

# Simpan model multitask
model.save("multitask_speaker2_model.h5")
print("Model multitask disimpan sebagai multitask_speaker_recognition_model.h5")

# Simpan encoder
import joblib
joblib.dump(command_encoder, "command_encoder2.pkl")
joblib.dump(speaker_encoder, "speaker_encoder2.pkl")
print("Encoder disimpan sebagai command_encoder2.pkl dan speaker_encoder2.pkl")

# Prediksi pada data testing
y_pred = model.predict(X_test)
y_commands_pred = np.argmax(y_pred[0], axis=1)
y_speakers_pred = np.argmax(y_pred[1], axis=1)

# Confusion matrix untuk command
cm_commands = confusion_matrix(y_commands_test, y_commands_pred)
cm_speakers = confusion_matrix(y_speakers_test, y_speakers_pred)

# Plot confusion matrix untuk command
plt.figure(figsize=(10, 8))
sns.heatmap(cm_commands, annot=True, fmt='d', cmap='Blues',
            xticklabels=command_encoder.classes_,
            yticklabels=command_encoder.classes_)
plt.title("Confusion Matrix for Command Recognition")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot confusion matrix untuk speaker
plt.figure(figsize=(10, 8))
sns.heatmap(cm_speakers, annot=True, fmt='d', cmap='Greens',
            xticklabels=speaker_encoder.classes_,
            yticklabels=speaker_encoder.classes_)
plt.title("Confusion Matrix for Speaker Recognition")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification report untuk command
print("Classification Report for Command Recognition:")
print(classification_report(y_commands_test, y_commands_pred, target_names=command_encoder.classes_))

# Classification report untuk speaker
print("Classification Report for Speaker Recognition:")
print(classification_report(y_speakers_test, y_speakers_pred, target_names=speaker_encoder.classes_))