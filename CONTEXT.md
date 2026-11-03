# liedetector — Project Context

## Ringkasan

Aplikasi web Django untuk latihan wawancara: pengguna menjawab pertanyaan di depan kamera, wajahnya
dianalisis (kedipan mata, senyum, arah pandang), dan sistem memberi perkiraan "probabilitas berbohong"
sebagai bahan evaluasi — bukan alat bukti.

## Cara kerja

Klien menangkap frame dari `getUserMedia` dan mengirimkannya ke server. Server memproses dengan
**OpenCV + dlib** memakai 68 titik penanda wajah:

- **Eye Aspect Ratio (EAR)** — rasio jarak kelopak; di bawah ambang `0.25` dihitung sebagai kedipan.
- **Intensitas senyum** — dari geometri mulut.
- **Arah pandang** — simpangan pupil terhadap titik tengah mata, disimpan sebagai `{'x', 'y'}`.
- `lie_probability` dihitung dari kombinasi ketiga sinyal di atas.

Hasil tiap sesi disimpan pada model `FaceAnalysisResult` (`user`, `timestamp`, `blink_count`,
`smile_intensity`, `gaze_direction`, `video_path`, `lie_probability`), sehingga riwayat bisa
dibandingkan antar sesi.

`calibrate.py` dipakai untuk menyesuaikan ambang pada kondisi kamera/pencahayaan setempat — wajib
dilakukan sebelum angka dianggap berarti.

## Struktur

```
manage.py                  Django
detector/analysis.py       LieDetector: deteksi wajah, EAR, senyum, pandangan
detector/calibrate.py      Penyetelan ambang
detector/models.py         FaceAnalysisResult
detector/views.py, urls.py Alur halaman dan endpoint
templates/                 home, question, camera
liedetector/               Settings dan root URL
```

## Menjalankan

```bash
pip install django opencv-python numpy scipy
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
python manage.py migrate
python manage.py runserver
```

Butuh `shape_predictor_68_face_landmarks.dat` di direktori kerja — tanpanya `LieDetector` gagal saat
inisialisasi.

## Catatan

- Akses kamera hanya di `localhost` atau HTTPS.
- Hasilnya indikator statistik, bukan penentu kebenaran; jangan dipakai untuk menuduh siapa pun.
- Wheel dlib di repo khusus CPython 3.10 Windows; platform lain perlu build dlib sendiri.
