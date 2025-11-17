# 🖋️ Klasifikasi Aksara Sasak dengan CNN

Project ini adalah sistem deep learning berbasis **Convolutional Neural Network (CNN)** untuk mendeteksi dan mengklasifikasikan **Aksara Sasak**. Program dapat melakukan prediksi untuk satu gambar maupun satu folder gambar sekaligus. Model dapat dijalankan melalui script Python maupun aplikasi web berbasis Streamlit.

---

## ✨ Fitur

- **Prediksi Single Image**  
  Mengklasifikasikan satu gambar aksara dengan menampilkan label dan confidence.

- **Prediksi Satu Folder**  
  Melakukan klasifikasi otomatis pada seluruh gambar dalam satu folder.

- **Top-3 Predictions**  
  Menampilkan tiga hasil prediksi terbaik beserta confidence masing-masing.

- **Export Hasil ke CSV**  
  Hasil prediksi bisa disimpan sebagai file `.csv`.

- **Aplikasi Streamlit**  
  Memudahkan pengguna untuk melakukan prediksi melalui antarmuka web sederhana.

---

## ⚙️ Persiapan

### 1. Clone Repository

```bash
git clone https://github.com/username/aksara-sasak.git
cd aksara-sasak
```

### 2. Clone Repository

```bash
pip install -r requirements.txt
```

### 🧠 Training Model
Jika file **model.h5** belum ada, jalankan:

```bash
python main.py
```

Script ini akan melakukan:

- Load dataset
- Preprocessing
- Augmentasi
- Training CNN
- Menyimpan hasil model ke models/model.h5

Jika **model.h5** sudah tersedia, langkah ini dapat dilewati.  


### 🌐 Menjalankan Streamlit
```bash
streamlit run apps.py
```


