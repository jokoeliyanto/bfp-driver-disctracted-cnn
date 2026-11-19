from django.conf import settings
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import ml, notebooks
from .forms import MAX_UPLOAD_MB, UploadForm

REPO_URL = "https://github.com/jokoeliyanto/bfp-driver-disctracted-cnn"

AUTHOR = {
    "name": "Joko Eliyanto",
    "role": "Data Scientist · Machine Learning · Bangkit 2020 graduate",
    "linkedin": "https://www.linkedin.com/in/joko-eliyanto-23a1b6143/",
    "github": "https://github.com/jokoeliyanto",
}
# Transcribed from certificate/BANGKIT.png
CERTIFICATE = {
    "title": "Certificate of Completion",
    "programme": "Bangkit 2020",
    "statement": "For successfully completing all of the requirements in Bangkit.",
    "issuer": "Bangkit, led by Google",
    "partners": "Google · Gojek · Tokopedia · Traveloka",
    "cohort": "Yogyakarta",
    "date": "September 9, 2020",
}
# From certificate/blog-news.txt and the article itself
ACHIEVEMENT = {
    "title": "One of the four best final projects of Bangkit 2020",
    "article": "Empat ide terbaik dari peserta Bangkit angkatan 2020",
    "article_en": "The four best ideas from the 2020 Bangkit cohort",
    "publisher": "Google Indonesia Blog",
    "date": "9 September 2020",
    "url": "https://blog.google/intl/id-id/company-news/outreach-initiatives/2020_09_empat-ide-terbaik-dari-peserta-bangki/",
    "project": "Distracted Driver Detector (Yogyakarta)",
    "quote": "Dalam membangun aplikasinya, kami memanfaatkan beberapa pelajaran yang diperoleh di Bangkit seperti "
             "Machine Learning, penerapan Dropout Layer yang dapat mengoptimalisasi model, dan Transfer Learning…",
    "quote_en": "To build the app we used what we learned in Bangkit: machine learning, dropout layers to optimise "
                "the model, and transfer learning…",
    "finalists": [
        ("Aksara Jawa", "Jakarta", "Learning app for Javanese script"),
        ("Distracted Driver Detector", "Yogyakarta", "This project"),
        ("Ingredients Classification", "Bandung", "Recipe ideas from a photo of ingredients"),
        ("Garbage Image Classification", "Denpasar", "Sorting recyclable waste"),
    ],
}
TEAM = [
    {"name": "Joko Eliyanto", "role": "Team member, Bangkit 2020 Yogyakarta"},
    {"name": "Muhammad Dany Alfikri", "role": "Team member, Bangkit 2020 Yogyakarta"},
    {"name": "Kadhana Reya Wisinggya", "role": "Team member, Bangkit 2020 Yogyakarta; hosts the published model",
     "url": "https://github.com/KadhReyA/state-distracted-cnn"},
]


def _common_context():
    metrics = ml.get_metrics()
    runs = {r["key"]: r for r in metrics["runs"]}
    return {"metrics": metrics, "deployed": runs["transfer"], "baseline": runs["baseline"], "repo_url": REPO_URL}


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def detect_view(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    result, sample = None, request.GET.get("sample")
    if request.method == "POST" and form.is_valid():
        try:
            result = ml.classify_bytes(form.cleaned_data["image"].read())
        except ml.InputError as exc:
            form.add_error("image", str(exc))
    elif sample:
        try:
            result = ml.classify_image(ml.load_sample(sample))
        except KeyError:
            raise Http404("Unknown sample image.")
        result["sample"] = sample
    if result:
        result["top"] = result["probabilities"][:3]
    return render(request, "detector/detect.html", {
        **_common_context(), "form": form, "result": result, "samples": ml.sample_names(),
        "classes": ml.classes(), "max_upload_mb": MAX_UPLOAD_MB})


def project_view(request):
    return render(request, "detector/project.html", {**_common_context(), "team": TEAM, "achievement": ACHIEVEMENT})


def evaluation_view(request):
    ctx = _common_context()
    metrics, runs = ctx["metrics"], ctx["metrics"]["runs"]
    chart = {r["key"]: {"name": r["name"], "epochs": [e["epoch"] for e in r["history"]],
                        **{k: [e[k] for e in r["history"]] for k in ("accuracy", "val_accuracy", "loss", "val_loss")}}
             for r in runs}
    evaluation = ml.get_evaluation()
    matrix = None
    if evaluation:
        codes = [c["code"] for c in metrics["classes"]]
        matrix = []
        for code, row in zip(codes, evaluation["confusion_matrix"]):
            total = sum(row) or 1
            matrix.append({"actual": code, "cells": [
                {"value": v, "share": v / total, "diagonal": code == pred} for pred, v in zip(codes, row)]})
    return render(request, "detector/evaluation.html", {
        **ctx, "chart": chart, "evaluation": evaluation, "matrix": matrix,
        "check": metrics["preprocessing_check"]})


def notebook_view(request):
    rendered = notebooks.render()
    return render(request, "detector/notebook.html", {
        **_common_context(), "toc": rendered["toc"], "cells": rendered["cells"], "code_cells": rendered["code"],
        "filename": rendered["filename"],
        "notebook_html": mark_safe(rendered["html"]),          # the repository's own notebook
        "highlight_css": mark_safe(notebooks.highlight_css()),
        "github_url": f"{REPO_URL}/blob/portfolio/notebooks/{rendered['filename']}",
    })


def notebook_download(request):
    path = settings.NOTEBOOK_PATH
    return FileResponse(open(path, "rb"), as_attachment=True, filename=path.name,
                        content_type="application/x-ipynb+json")


def certification_view(request):
    return render(request, "detector/certification.html", {
        **_common_context(), "author": AUTHOR, "certificate": CERTIFICATE, "achievement": ACHIEVEMENT, "team": TEAM})


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@csrf_exempt
@require_POST
def api_predict(request):
    """Classify one dashboard photo.

    Send it as multipart form data in a field called "image", or as the raw request body:
        curl -F image=@driver.jpg http://127.0.0.1:8000/api/predict/
    """
    upload = request.FILES.get("image")
    data = upload.read() if upload else request.body
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return JsonResponse({"error": f"Images are limited to {MAX_UPLOAD_MB} MB."}, status=400)
    try:
        result = ml.classify_bytes(data)
    except ml.InputError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    result.pop("preview")
    return JsonResponse({**result, "model": ml.get_metrics()["model_name"]})
