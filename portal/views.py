import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt

from .models import Student


KNOWN_FACES_DIR = Path(settings.BASE_DIR) / 'portal' / 'known_faces'
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def decode_base64_image(data_url):
    if not data_url:
        return None

    if ',' in data_url:
        _, encoded = data_url.split(',', 1)
    else:
        encoded = data_url

    try:
        image_bytes = base64.b64decode(encoded)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception:
        return None


def detect_faces(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    return faces


def extract_face_vector_from_camera(image_bgr):
    faces = detect_faces(image_bgr)

    if len(faces) == 0:
        return None, "Yuz aniqlanmadi."

    if len(faces) > 1:
        return None, "Kadrda faqat bitta yuz bo‘lishi kerak."

    x, y, w, h = faces[0]
    face = image_bgr[y:y+h, x:x+w]

    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_face, (120, 120))
    normalized = resized.astype("float32") / 255.0
    vector = normalized.flatten()

    return vector, None


def extract_face_vector_from_saved_face(face_image_bgr):
    try:
        gray_face = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_face, (120, 120))
        normalized = resized.astype("float32") / 255.0
        vector = normalized.flatten()
        return vector, None
    except Exception:
        return None, "Saqlangan rasmni qayta ishlashda xatolik."


def compare_with_known_faces(input_vector):
    students = Student.objects.all()
    if not students.exists():
        return None, None

    best_student = None
    best_score = float("inf")

    for student in students:
        image_path = KNOWN_FACES_DIR / student.image_file

        if not image_path.exists():
            continue

        known_image = cv2.imread(str(image_path))
        if known_image is None:
            continue

        known_vector, error = extract_face_vector_from_saved_face(known_image)
        if error is not None or known_vector is None:
            continue

        distance = np.linalg.norm(input_vector - known_vector)

        if distance < best_score:
            best_score = distance
            best_student = student

    return best_student, best_score


def generate_student_id():
    last_student = Student.objects.order_by('-id').first()

    if last_student:
        try:
            last_number = int(last_student.sid.split('-')[-1])
        except Exception:
            last_number = last_student.id
        new_number = last_number + 1
    else:
        new_number = 1

    return f"ST-{new_number:05d}"


def login_page(request):
    if request.session.get('verified_student'):
        return redirect('dashboard')
    return render(request, 'login.html')


def register_page(request):
    students = Student.objects.all().order_by('id')
    return render(request, 'register.html', {'students': students})


@csrf_exempt
def add_student(request):
    if request.method != 'POST':
        return JsonResponse(
            {'ok': False, 'error': 'Faqat POST ruxsat etiladi.'},
            status=405
        )

    name = request.POST.get('name', '').strip()
    image_data = request.POST.get('image', '').strip()

    if not name:
        return JsonResponse(
            {'ok': False, 'error': 'Talaba ismi kiritilmadi.'},
            status=400
        )

    if Student.objects.filter(name=name).exists():
        return JsonResponse(
            {'ok': False, 'error': 'Bu ism bilan talaba allaqachon mavjud.'},
            status=400
        )

    image_bgr = decode_base64_image(image_data)
    if image_bgr is None:
        return JsonResponse(
            {'ok': False, 'error': 'Rasm o‘qilmadi.'},
            status=400
        )

    faces = detect_faces(image_bgr)

    if len(faces) == 0:
        return JsonResponse(
            {'ok': False, 'error': 'Yuz aniqlanmadi.'},
            status=400
        )

    if len(faces) > 1:
        return JsonResponse(
            {'ok': False, 'error': 'Kadrda faqat bitta yuz bo‘lishi kerak.'},
            status=400
        )

    x, y, w, h = faces[0]
    face = image_bgr[y:y+h, x:x+w]
    resized_face = cv2.resize(face, (300, 300))

    safe_name = "".join(
        ch for ch in name if ch.isalnum() or ch in (' ', '_', '-')
    ).strip()

    if not safe_name:
        return JsonResponse(
            {'ok': False, 'error': 'Ism noto‘g‘ri formatda.'},
            status=400
        )

    file_name = f"{safe_name}.jpg"
    save_path = KNOWN_FACES_DIR / file_name

    success = cv2.imwrite(str(save_path), resized_face)
    if not success:
        return JsonResponse(
            {'ok': False, 'error': 'Rasm faylga saqlanmadi.'},
            status=500
        )

    student = Student.objects.create(
        name=name,
        sid=generate_student_id(),
        image_file=file_name
    )

    return JsonResponse({
        'ok': True,
        'message': f'{student.name} muvaffaqiyatli saqlandi.',
        'sid': student.sid
    })


@csrf_exempt
def verify_face(request):
    if request.method != 'POST':
        return JsonResponse(
            {'ok': False, 'error': 'Faqat POST ruxsat etiladi.'},
            status=405
        )

    image_data = request.POST.get('image', '').strip()
    image_bgr = decode_base64_image(image_data)

    if image_bgr is None:
        return JsonResponse(
            {'ok': False, 'error': 'Rasm noto‘g‘ri yuborildi.'},
            status=400
        )

    input_vector, error = extract_face_vector_from_camera(image_bgr)
    if error is not None or input_vector is None:
        return JsonResponse(
            {'ok': False, 'error': error},
            status=400
        )

    matched_student, score = compare_with_known_faces(input_vector)

    if matched_student is None:
        return JsonResponse(
            {'ok': False, 'error': 'Bazadagi talabalar bilan solishtirib bo‘lmadi.'},
            status=400
        )

    threshold = 20.0

    if score is None or score > threshold:
        return JsonResponse(
            {
                'ok': False,
                'error': 'Talaba aniqlanmadi.',
                'score': None if score is None else round(float(score), 2)
            },
            status=401
        )

    request.session['verified_student'] = True
    request.session['student_name'] = matched_student.name
    request.session['student_id'] = matched_student.sid

    return JsonResponse({
        'ok': True,
        'name': matched_student.name,
        'sid': matched_student.sid,
        'score': round(float(score), 2)
    })


def dashboard(request):
    if not request.session.get('verified_student'):
        return redirect('login')

    context = {
        'name': request.session.get('student_name'),
        'sid': request.session.get('student_id'),
    }
    return render(request, 'dashboard.html', context)


def logout_view(request):
    request.session.flush()
    return redirect('login')
