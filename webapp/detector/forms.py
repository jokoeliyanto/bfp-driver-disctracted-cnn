from django import forms

MAX_UPLOAD_MB = 5


class UploadForm(forms.Form):
    image = forms.FileField(
        label="Dashboard camera photo",
        help_text=f"PNG, JPEG or WebP, up to {MAX_UPLOAD_MB} MB. Best results come from a camera on the passenger "
                  "side, looking at the driver, like the State Farm photos.",
        widget=forms.ClearableFileInput(attrs={"accept": "image/png,image/jpeg,image/webp"}))

    def clean_image(self):
        f = self.cleaned_data["image"]
        if f.size > MAX_UPLOAD_MB * 1024 * 1024:
            raise forms.ValidationError(f"Please upload an image smaller than {MAX_UPLOAD_MB} MB.")
        return f
