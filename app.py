import os
import time
import uuid
from io import BytesIO
from pathlib import Path

from flask import Flask, Response, render_template, request
from PIL import Image, ImageOps
from rembg import new_session, remove
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_UPLOAD_SIZE = 30 * 1024 * 1024
MAX_PROCESS_PIXELS = 12_000_000
RESULTS: dict[str, tuple[bytes, str, str, float]] = {}
MAX_RESULT_CACHE_ITEMS = 3

# Allow the full configured upload size range. The default Pillow pixel cap is too small
# for large images and can block valid uploads before the background-removal logic begins.
Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
UPLOAD_FOLDER.mkdir(exist_ok=True)
REMBG_SESSION = new_session("u2netp", providers=["CPUExecutionProvider"])


def cleanup_old_files(folder: Path, max_age_hours: int = 24) -> None:
    """Delete stale files so the app does not accumulate uploads and outputs."""
    cutoff = time.time() - (max_age_hours * 3600)
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.stat().st_mtime < cutoff:
            file_path.unlink(missing_ok=True)


@app.before_request
def cleanup_stale_files() -> None:
    cleanup_old_files(UPLOAD_FOLDER)
    cleanup_old_results()


def cleanup_old_results(max_age_hours: int = 1) -> None:
    """Delete stale in-memory results to avoid unbounded memory growth."""
    cutoff = time.time() - (max_age_hours * 3600)
    for result_id, (_, _, _, created_at) in list(RESULTS.items()):
        if created_at < cutoff:
            RESULTS.pop(result_id, None)


def get_uploaded_image() -> tuple[Image.Image | None, str | None]:
    file = request.files.get("image")
    if not file or not file.filename:
        return None, "Please choose an image file."

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return None, "Unsupported file type. Please upload a JPG, PNG, WEBP, or BMP image."

    try:
        image = ImageOps.exif_transpose(Image.open(file))
        image.load()
        return image, None
    except Exception:
        return None, "The image could not be opened. Please try another image."


def output_filename(original_name: str, extension: str) -> str:
    safe_name = secure_filename(original_name)
    stem = Path(safe_name).stem or uuid.uuid4().hex
    return f"{stem}{extension}"


def store_result(data: bytes, mimetype: str, filename: str) -> str:
    result_id = uuid.uuid4().hex
    RESULTS[result_id] = (data, mimetype, filename, time.time())

    if len(RESULTS) > MAX_RESULT_CACHE_ITEMS:
        oldest_id = min(RESULTS, key=lambda item: RESULTS[item][3])
        RESULTS.pop(oldest_id, None)

    return result_id


def prepare_input_for_removal(image: Image.Image) -> Image.Image:
    width, height = image.size
    pixel_count = width * height

    if pixel_count <= MAX_PROCESS_PIXELS:
        return image

    scale = (MAX_PROCESS_PIXELS / pixel_count) ** 0.5
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buffer.getvalue()


def resize_image(image: Image.Image, scale: float) -> Image.Image:
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def add_noise_for_size(image: Image.Image, strength: float = 0.04) -> Image.Image:
    noise = Image.effect_noise(image.size, 16).convert("RGB")
    return Image.blend(image.convert("RGB"), noise, strength)


def search_quality_for_target(image: Image.Image, target_bytes: int) -> bytes:
    best = jpeg_bytes(image, 95)

    low, high = 10, 95
    for _ in range(8):
        quality = (low + high) // 2
        encoded = jpeg_bytes(image, quality)
        if len(encoded) <= target_bytes:
            best = encoded
            low = quality + 1
        else:
            high = quality - 1

    return best


def fit_jpeg_target(image: Image.Image, target_bytes: int, minimum_bytes: int = 2_000) -> bytes:
    image = image.convert("RGB")
    target_bytes = max(1, target_bytes)
    original_bytes = len(jpeg_bytes(image, 95))

    # Reduction path: keep the current image and lower quality until the requested size is met.
    if target_bytes <= original_bytes:
        candidate = image
        scale = 1.0

        for _ in range(10):
            encoded = search_quality_for_target(candidate, target_bytes)
            encoded_size = len(encoded)
            if encoded_size <= target_bytes:
                return encoded

            scale *= 0.85
            if scale < 0.14:
                break

            candidate = resize_image(image, scale)

        return search_quality_for_target(candidate, target_bytes)

    # Increase path: prefer lightweight noise on the original image first, then modest
    # upscales only if that is still not enough. This avoids the huge memory spikes from
    # blindly resizing to the square root of the requested byte size.
    best_bytes = original_bytes
    best_candidate = image
    noise_steps = (0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40)

    for strength in noise_steps:
        noisy = add_noise_for_size(image, strength)
        encoded = jpeg_bytes(noisy, 95)
        encoded_size = len(encoded)
        if encoded_size >= target_bytes:
            return encoded
        if encoded_size > best_bytes:
            best_bytes = encoded_size
            best_candidate = noisy

    for scale in (1.25, 1.50, 1.75, 2.00, 2.50, 3.00):
        scaled = resize_image(image, scale)
        for strength in noise_steps:
            noisy = add_noise_for_size(scaled, strength)
            encoded = jpeg_bytes(noisy, 95)
            encoded_size = len(encoded)
            if encoded_size >= target_bytes:
                return encoded
            if encoded_size > best_bytes:
                best_bytes = encoded_size
                best_candidate = noisy

    # If the requested size is still not achievable safely, return the best candidate we found.
    return jpeg_bytes(best_candidate, 95)


def pad_bytes_to_target(data: bytes, target_bytes: int) -> bytes:
    if len(data) >= target_bytes:
        return data[:target_bytes]
    return data + (b"\0" * (target_bytes - len(data)))


def parse_target_size(field: str, minimum: int, maximum: int) -> int | None:
    try:
        value = int(request.form.get(field, "")) * 1024
    except ValueError:
        return None
    return value if minimum <= value <= maximum else None


@app.route("/", methods=["GET", "POST"])
def index():
    result_id = None
    output_file_name = None
    error = None
    uploaded_name = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or not file.filename:
            error = "Please choose an image file to remove the background."
            return render_template("index.html", output_data_url=None, error=error)

        extension = os.path.splitext(file.filename)[1].lower()
        uploaded_name = secure_filename(file.filename) or file.filename
        if extension not in ALLOWED_EXTENSIONS:
            error = "Unsupported file type. Please upload a JPG, PNG, WEBP, or BMP image."
            return render_template("index.html", output_data_url=None, error=error)

        input_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}{extension}"

        try:
            file.save(input_path)

            with Image.open(input_path) as input_img:
                input_img = ImageOps.exif_transpose(input_img)
                input_img = prepare_input_for_removal(input_img)

                if input_img.mode not in ("RGB", "RGBA"):
                    input_img = input_img.convert("RGBA")

                output_img = remove(
                    input_img,
                    session=REMBG_SESSION,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_size=10,
                    post_process_mask=True,
                )

                if output_img.mode != "RGBA":
                    output_img = output_img.convert("RGBA")

                buffer = BytesIO()
                output_img.save(
                    buffer,
                    format="PNG",
                    compress_level=6,
                    optimize=True,
                )
                output_bytes = buffer.getvalue()

            output_file_name = output_filename(file.filename, ".png")
            result_id = store_result(output_bytes, "image/png", output_file_name)

        except Exception as exc:
            app.logger.exception("Error processing uploaded image")
            error = "The image could not be processed. Please try another image."
            if app.debug:
                error = f"{error} Details: {exc}"
            return render_template("index.html", output_data_url=None, error=error)

        finally:
            if input_path.exists():
                input_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        result_id=result_id,
        output_file_name=output_file_name,
        uploaded_name=uploaded_name,
        error=error,
    )


def process_standard_image(operation: str):
    image, error = get_uploaded_image()
    uploaded_file = request.files.get("image")
    original_name = secure_filename(uploaded_file.filename) if uploaded_file else "image"
    if error:
        return render_template("index.html", error=error, active_tool=operation)

    try:
        if operation == "increase":
            target = parse_target_size("target_size", 1 * 1024, MAX_UPLOAD_SIZE)
            if target is None:
                raise ValueError
            resized_bytes = fit_jpeg_target(image, target, 10 * 1024)
            output_bytes = pad_bytes_to_target(resized_bytes, target)
            output_file_name = output_filename(original_name, ".jpg")
            output_mimetype = "image/jpeg"
            message = "Image enlarged to the requested file size."
        elif operation == "reduce":
            target = parse_target_size("target_size", 1 * 1024, MAX_UPLOAD_SIZE)
            if target is None:
                raise ValueError
            output_bytes = fit_jpeg_target(image, target)
            output_file_name = output_filename(original_name, ".jpg")
            output_mimetype = "image/jpeg"
            message = "Image compressed to the requested file size."
        else:
            ratio_parts = request.form.get("ratio", "1:1").split(":", 1)
            width = int(ratio_parts[0])
            height = int(ratio_parts[1])
            if width < 1 or height < 1 or width > 100 or height > 100:
                raise ValueError
            ratio = width / height
            target_width = image.width
            target_height = max(1, round(target_width / ratio))
            if request.form.get("ratio_mode", "crop") == "fit":
                fitted = ImageOps.contain(image.convert("RGBA"), (target_width, target_height))
                canvas = Image.new("RGBA", (target_width, target_height), (255, 255, 255, 255))
                canvas.alpha_composite(
                    fitted,
                    ((target_width - fitted.width) // 2, (target_height - fitted.height) // 2),
                )
                image = canvas.convert("RGB")
            else:
                image = ImageOps.fit(image, (target_width, target_height), method=Image.Resampling.LANCZOS)
            output_bytes = jpeg_bytes(image, 95)
            output_file_name = output_filename(original_name, ".jpg")
            output_mimetype = "image/jpeg"
            message = "Image ratio adjusted successfully."
    except (ValueError, OSError):
        return render_template("index.html", error="Please check the selected settings and try again.", active_tool=operation)

    result_id = store_result(output_bytes, output_mimetype, output_file_name)

    return render_template(
        "index.html",
        result_id=result_id,
        output_file_name=output_file_name,
        uploaded_name=original_name,
        message=message,
        active_tool=operation,
    )


@app.route("/increase-size", methods=["POST"])
def increase_size():
    return process_standard_image("increase")


@app.route("/reduce-size", methods=["POST"])
def reduce_size():
    return process_standard_image("reduce")


@app.route("/adjust-ratio", methods=["POST"])
def adjust_ratio():
    return process_standard_image("ratio")


@app.route("/result/<result_id>")
def result_image(result_id: str):
    result = RESULTS.get(result_id)
    if result is None:
        return "Result not found", 404

    data, mimetype, _, _ = result
    return Response(data, mimetype=mimetype)


@app.route("/download/<result_id>")
def download_result(result_id: str):
    result = RESULTS.get(result_id)
    if result is None:
        return "Result not found", 404

    data, mimetype, filename, _ = result
    return Response(
        data,
        mimetype=mimetype,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/health")
def health_check():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
