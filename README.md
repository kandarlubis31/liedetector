# liedetector

A Django web app that watches your webcam and reports blink-rate and gaze statistics while
you answer questions.

## What this is, and what it is not

Despite the name, this does **not** detect lies. There is no scientific basis for inferring
deception from blink rate, and real polygraphs are not reliable either.

What it actually does is computer vision: it locates facial landmarks and measures the
**Eye Aspect Ratio (EAR)** to count blinks and estimate where you are looking. Any
"probability" shown is a heuristic on top of those measurements, useful only as a
demonstration of the technique.

Treat it as a CV experiment, not a truth machine.

## How it works

- `detector/analysis.py` uses **dlib** (68-point landmark predictor) and computes EAR from
  the six eye landmarks per side. A blink is counted when the ratio drops below `0.25`.
- `detector/views.py` uses **MediaPipe Face Mesh** for a second, landmark-only pipeline and
  streams annotated frames to the browser as MJPEG.
- There is a **10-second calibration phase** before measurement starts, so the baseline is
  taken from the person in front of the camera rather than a fixed constant.
- Analysis runs for a minimum of **15 seconds**; blink history is kept in a short deque so
  the score reacts to recent behaviour instead of the whole session.
- Templates in `templates/` provide the home page, the question flow and the camera view.

## Requirements

- Python 3.10
- A webcam

`dlib` needs CMake and a C++ compiler, or the prebuilt wheel included in the repository
(`dlib-19.22.99-cp310-cp310-win_amd64.whl`) — it is Windows + Python 3.10 only.

The landmark model `shape_predictor_68_face_landmarks.dat` is **not** included; download it
from the dlib model page and place it in the project root.

## Setup

```bash
pip install -r requirements.txt
# Windows / Python 3.10 shortcut:
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl

python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## Structure

```
detector/
  analysis.py      EAR blink detection with dlib
  calibrate.py     baseline calibration helpers
  views.py         MJPEG stream + JSON results (MediaPipe)
  models.py
  templates/       UI
liedetector/       Django project (settings, urls, wsgi/asgi)
templates/         home, question, camera pages
```

## Notes

- `db.sqlite3` is committed. It contains only Django's own auth/migration tables with zero
  users and zero sessions, so nothing personal is in it — but it is still a database file
  in a repo.
- Video never leaves the machine; frames are processed locally.

---

## Bahasa Indonesia

**liedetector** adalah aplikasi web Django yang memantau webcam dan melaporkan statistik kedipan mata serta arah pandangan selama pengguna menjawab pertanyaan. README-nya menjelaskan secara terbuka apa yang alat ini bisa dan tidak bisa simpulkan.

---

## License

MIT
