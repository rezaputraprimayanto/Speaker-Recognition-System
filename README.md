**PENERAPAN CONVOLUTIONAL NEURAL NETWORK UNTUK PENGENALAN PEMBICARA SEBAGAI SISTEM KENDALI PINTU BERBASIS RASPBERRY PI 5**

**Edge Intelligent and Computing**

Anggota Kelompok :
1. Reza Putra Primayanto (215150301111031)
2. Aufada Muflih Sastropawiro (215150300111003)
3. Takeru Iwasawa (215150300111040)
4. Khrisna Shane Budy Prakoso (215150300111010)
5. Ardhito Hafiz Prathama (215150300111005)

**Project Domain**
Proyek ini berfokus pada implementasi pengenalan suara berbasis kecerdasan buatan untuk sistem kendali pintu otomatis. Sistem ini memanfaatkan fitur MFCC (Mel-Frequency Cepstral Coefficients) untuk ekstraksi ciri suara dan CNN (Convolutional Neural Network) untuk pengenalan suara yang aman

**Problem Statements**
1. Sistem kunci pintu tradisional rentan terhadap risiko keamanan, seperti kehilangan kunci fisik atau pembobolan.
2. Tidak adanya sistem autentikasi berbasis suara yang mudah diakses dan dapat diimplementasikan dengan perangkat keras sederhana.
3. Sistem kendali pintu modern membutuhkan solusi yang aman, praktis, dan efisien.

**Goals**
1. Mengembangkan sistem kendali pintu berbasis suara yang hanya dapat diakses oleh pengguna terdaftar.
2. Memanfaatkan teknologi AI untuk meningkatkan keamanan dengan pengenalan suara.
3. Menyediakan solusi yang terjangkau dan mudah diterapkan menggunakan perangkat keras seperti Raspberry Pi 5.

**Solution Statements**
1. **Ekstraksi Fitur:** Menggunakan MFCC untuk mengekstraksi karakteristik unik dari suara pengguna.
2. **Klasifikasi Suara:** Melatih dan menggunakan CNN untuk membedakan suara pengguna yang diizinkan dan tidak diizinkan.
3. **Integrasi Hardware:** Menghubungkan Raspberry Pi 5 dengan mikrofon USB untuk akuisisi suara dan solenoid lock untuk penguncian pintu.
4. **Respons Sistem:** Memberikan notifikasi (visual) untuk memastikan tindakan berhasil dilakukan.

**Prerequisites Hardware Components**
1. **Raspberry Pi 5:** Prosesor utama untuk pengolahan suara dan pengendalian perangkat keras.
2. **USB Condenser Microphone:** Menangkap suara pengguna dengan kualitas tinggi.
3. **Solenoid Lock:** Aktuator untuk membuka dan mengunci pintu.
4. **Relay Module:** Mengontrol daya ke solenoid lock.
5. **Power Supply:** Untuk Raspberry Pi dan solenoid lock.

**Software Requirements**
Python 3.x: Bahasa pemrograman utama untuk pengembangan sistem.
**Libraries:**
1. **Librosa:** Ekstraksi fitur MFCC dari audio.
2. **TensorFlow/Keras:** Membuat dan melatih model CNN.
3. **PySerial:** Mengontrol relay jika diperlukan.
4. **Dataset:** Kumpulan data suara untuk pelatihan dan pengujian model pengenalan suara.

**Flowchart Sistem**
![EDGE4 drawio](https://github.com/user-attachments/assets/6f3bea63-a45d-49f3-b235-94fb9f2fa5e1)

**How It Works**
1. **Input**: Suara pengguna diterima melalui USB condenser microphone.
2. **Feature Extraction**: Menggunakan MFCC untuk mengekstrak fitur suara.
3. **Classification**: Model CNN memproses data suara untuk mengenali apakah suara berasal dari pengguna terdaftar.
4. **Action**: Jika suara terverifikasi, Raspberry Pi mengaktifkan solenoid lock melalui relay untuk membuka pintu.




