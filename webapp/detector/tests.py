import io

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase
from django.urls import reverse
from PIL import Image

from . import ml


def _image_bytes(size=(640, 480), color=(90, 90, 90), mode="RGB", fmt="PNG"):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt)
    return buf.getvalue()


def _sample_bytes(i=0):
    return (settings.SAMPLES_DIR / ml.sample_names()[i]).read_bytes()


class ModelTests(SimpleTestCase):
    def test_weights_match_notebook_architecture(self):
        metrics = ml.get_metrics()
        transfer = next(r for r in metrics["runs"] if r["key"] == "transfer")
        self.assertEqual(ml.get_model().count_params(), transfer["params"]["total"])

    def test_class_order_and_safe_class(self):
        codes = [c["code"] for c in ml.classes()]
        self.assertEqual(codes, [f"c{i}" for i in range(10)])
        self.assertEqual([c["code"] for c in ml.classes() if not c["distracted"]], ["c0"])
        labels = (settings.PROJECT_DIR / "models" / "labels.txt").read_text().splitlines()
        self.assertEqual(labels[0].lower(), "safe driving")
        self.assertEqual(labels[6].lower(), "drinking")

    def test_raw_pixels_give_varied_predictions(self):
        # The published weights expect 0-255 input; rescaled input collapses to a single class
        check = ml.get_metrics()["preprocessing_check"]
        self.assertGreater(check["raw_0_255"]["distinct_classes"], 1)
        self.assertEqual(check["rescaled_0_1"]["distinct_classes"], 1)
        codes = {ml.classify_bytes(_sample_bytes(i))["code"] for i in range(len(ml.sample_names()))}
        self.assertEqual(len(codes), check["raw_0_255"]["distinct_classes"])

    def test_result_structure(self):
        result = ml.classify_bytes(_sample_bytes())
        self.assertAlmostEqual(sum(p["probability"] for p in result["probabilities"]), 1.0, places=4)
        self.assertAlmostEqual(result["p_safe"] + result["p_distracted"], 1.0, places=6)
        self.assertEqual(result["distracted"], result["code"] != "c0")
        self.assertEqual(result["code"], result["probabilities"][0]["code"])
        self.assertTrue(result["preview"].startswith("data:image/jpeg;base64,"))

    def test_accepts_other_formats_and_modes(self):
        for kwargs in ({"mode": "RGBA", "color": (90, 90, 90, 255)}, {"mode": "L", "color": 128},
                       {"fmt": "JPEG"}, {"size": (3000, 2000)}, {"size": (50, 40)}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                self.assertIn("code", ml.classify_bytes(_image_bytes(**kwargs)))

    def test_rejects_invalid_files(self):
        for data in (b"", b"not an image", b"%PDF-1.4 fake"):
            with self.subTest(data=data), self.assertRaises(ml.InputError):
                ml.classify_bytes(data)


class ViewTests(SimpleTestCase):
    def test_pages_render(self):
        for name in ("detect", "project", "evaluation", "notebook", "certification"):
            with self.subTest(page=name):
                self.assertEqual(self.client.get(reverse(f"detector:{name}")).status_code, 200)

    def test_upload_shows_verdict(self):
        resp = self.client.post(reverse("detector:detect"),
                                {"image": SimpleUploadedFile("driver.jpg", _sample_bytes(1), "image/jpeg")})
        self.assertContains(resp, 'role="status"')
        self.assertTrue(resp.context["result"]["top"])

    def test_upload_rejects_non_images(self):
        resp = self.client.post(reverse("detector:detect"),
                                {"image": SimpleUploadedFile("notes.txt", b"hello", "text/plain")})
        self.assertContains(resp, "not a readable image")
        self.assertIsNone(resp.context["result"])

    def test_sample_and_unknown_sample(self):
        name = ml.sample_names()[0]
        resp = self.client.get(reverse("detector:detect"), {"sample": name})
        self.assertEqual(resp.context["result"]["sample"], name)
        self.assertEqual(self.client.get(reverse("detector:detect"), {"sample": "../settings.py"}).status_code, 404)

    def test_evaluation_without_full_results(self):
        resp = self.client.get(reverse("detector:evaluation"))
        if ml.get_evaluation() is None:
            self.assertContains(resp, "scripts/evaluate.py")
        self.assertIn("transfer", resp.context["chart"])

    def test_notebook_has_sections_and_download(self):
        resp = self.client.get(reverse("detector:notebook"))
        self.assertGreaterEqual(len(resp.context["toc"]), 8)
        download = self.client.get(reverse("detector:notebook_download"))
        self.assertEqual(download.status_code, 200)
        self.assertIn("attachment", download["Content-Disposition"])

    def test_certification_links_achievement(self):
        resp = self.client.get(reverse("detector:certification"))
        self.assertContains(resp, "blog.google/intl/id-id")
        self.assertContains(resp, "bangkit-top4-blog.jpg")
        self.assertContains(resp, "bangkit-certificate.jpg")

    def test_api_multipart_and_raw_body(self):
        url = reverse("detector:api_predict")
        data = _sample_bytes()
        multipart = self.client.post(url, {"image": SimpleUploadedFile("x.jpg", data, "image/jpeg")}).json()
        raw = self.client.post(url, data, content_type="image/jpeg").json()
        self.assertEqual(multipart["code"], raw["code"])
        self.assertEqual(len(raw["probabilities"]), 10)
        self.assertNotIn("preview", raw)

    def test_api_errors(self):
        url = reverse("detector:api_predict")
        self.assertEqual(self.client.post(url, b"nope", content_type="image/png").status_code, 400)
        self.assertEqual(self.client.get(url).status_code, 405)
