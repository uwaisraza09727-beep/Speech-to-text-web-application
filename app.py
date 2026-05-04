from werkzeug.security import check_password_hash, generate_password_hash
from flask import Flask, render_template, request, redirect, session
import whisper
from transformers import pipeline
import os
from pymongo import MongoClient
from functools import wraps
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'webm', 'ogg'}
from flask import send_file
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import subprocess
import datetime


app = Flask(__name__)
# Use env variable for secret key
app.secret_key = os.environ.get("SECRET_KEY", "secret123")


app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Auto create uploads folder
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    

# Use env variable for MongoDB URI 
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI environment variable is not set. "
        "Add it as a HuggingFace Space secret."
    )
client = MongoClient(MONGO_URI)
db = client["speech_app"]
users_collection = db["users"]

# Lazy-load models to avoid HF Spaces startup timeout
_whisper_model = None
_sentiment_pipeline = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("small")
    return _whisper_model

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline("sentiment-analysis")
    return _sentiment_pipeline


def trim_audio(input_path, output_path):
    """Convert audio to 16kHz mono WAV — no hard 10s cut so full speech is captured."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ac", "1",        # mono
            "-ar", "16000",    # 16 kHz — Whisper's native rate
            "-f", "wav",       # force WAV container so Whisper reads it correctly
            output_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


# Middleware: login required
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper

# admin_required now also checks login
def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return redirect("/login")
        if 'role' not in session or session['role'] != 'admin':
            return "Access Denied! Admins only", 403
        return f(*args, **kwargs)
    return wrapper

# Home route
@app.route("/")
def home():
    return redirect("/login")

# Upload Page (protected)
@app.route("/upload-page")
@login_required
def upload_page():
    return render_template("index.html")

# Upload + Speech to Text
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if 'audio' not in request.files:
        return "No file selected!"

    file = request.files['audio']

    if file.filename == '':
        return "No file selected!"

    if not allowed_file(file.filename):
        return "Invalid file type!"

    if not allowed_mime(file):
        return "Invalid file content!"

    # language selection
    language = request.form.get("language") or "auto"

    # secure filename
    filename = secure_filename(file.filename)

    upload_folder = os.path.abspath(app.config["UPLOAD_FOLDER"])
    filepath = os.path.abspath(os.path.join(upload_folder, filename))

    # path traversal protection
    if not filepath.startswith(upload_folder):
        return "Path traversal detected!"

    # save file
    file.save(filepath)

    print("File received:", file.filename)

    # Always write converted audio as .wav so Whisper can decode it without ambiguity
    base_name = os.path.splitext(filename)[0]
    trimmed_path = os.path.join(upload_folder, "trimmed_" + base_name + ".wav")

    try:
        trim_audio(filepath, trimmed_path)
        filepath = trimmed_path
        print("Audio converted to WAV:", trimmed_path)
    except Exception as e:
        print("Audio conversion failed (using original):", e)
        # filepath stays as the original — Whisper will attempt it directly

    # default values
    text = "Transcription failed"
    sentiment_result = [{"label": "N/A", "score": 0}]

    try:
        model = get_whisper_model()
        sentiment = get_sentiment_pipeline()

        if language == "auto":
            result = model.transcribe(
                filepath,
                fp16=False,
                beam_size=5        # beam search; best_of only applies when beam_size=1
            )
            language = result.get("language", "unknown")
        else:
            result = model.transcribe(
                filepath,
                fp16=False,
                language=language, # actually pass the chosen language to Whisper
                beam_size=5
            )
            language = result.get("language", language)

        text = result["text"].strip()

        if not text:
            text = "No speech detected"

        # Sentiment models have a 512-token limit; truncate to avoid silent wrong results
        sentiment_input = text[:1000]
        sentiment_result = sentiment(sentiment_input, truncation=True, max_length=512)

    except Exception as e:
        import traceback
        print("Transcription/sentiment error:", traceback.format_exc())
        text = f"Transcription failed: {str(e)}"

    # store history with consistent field names
    history_collection = db["history"]
    history_collection.insert_one({
        "user_email": session['user'],   # consistent key
        "filename": filename,
        "file_path": filepath,
        "transcription": text,
        "sentiment": {
            "label": sentiment_result[0]['label'],
            "score": float(sentiment_result[0]['score'])
        },
        "language": language,
        "duration": 10,
        "created_at": datetime.datetime.utcnow()   # consistent key
    })

    return render_template(
        "index.html",
        transcription=text,
        sentiment=sentiment_result,
        language=language
    )

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email'].strip().lower()
        password = generate_password_hash(request.form['password'])

        #  duplicate-check and insert must be INSIDE the POST block
        if users_collection.find_one({"email": email}):
            return "User already exists!"

        users_collection.insert_one({
            "name": name,
            "email": email,
            "password": password,
            "role": "user",
            "created_at": datetime.datetime.utcnow(),
            "last_login": None,
            "is_active": True
        })

        return redirect("/login")

    return render_template('register.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        user = users_collection.find_one({"email": email})


        if user and check_password_hash(user['password'], password):
            users_collection.update_one(
                {"email": email},
                {"$set": {"last_login": datetime.datetime.utcnow()}}
            )

            session['user'] = email
            session['role'] = user.get('role', 'user')
            return redirect("/upload-page")
        else:
            return "Invalid Email or Password!"
    return render_template('login.html')

# Admin Route
@app.route('/admin')
@admin_required
def admin_dashboard():
    users = list(users_collection.find({}, {"password": 0}))
    return render_template("admin.html", users=users)

# Delete User Route
@app.route('/delete-user/<user_id>')
@admin_required
def delete_user(user_id):
    users_collection.delete_one({"_id": ObjectId(user_id)})
    return redirect("/admin")

# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect("/login")

# txt download route
@app.route('/download-txt')
@login_required
def download_txt():
    text = request.args.get("text", "")

    return send_file(
        io.BytesIO(text.encode()),
        as_attachment=True,
        download_name="transcription.txt",
        mimetype="text/plain"
    )

# pdf download route
@app.route('/download-pdf')
@login_required
def download_pdf():
    text = request.args.get("text", "")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [Paragraph(text, styles["Normal"])]
    doc.build(content)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="transcription.pdf",
        mimetype="application/pdf"
    )

# history route — use correct field names matching insert
@app.route('/history')
@login_required
def history():
    history_collection = db["history"]

    page = int(request.args.get("page", 1))
    per_page = 5

    skip = (page - 1) * per_page

    # query by "user_email" (matches insert), sort by "created_at" (matches insert)
    total = history_collection.count_documents({"user_email": session['user']})

    data = list(history_collection.find(
        {"user_email": session['user']},
        {"_id": 0}
    ).sort("created_at", -1).skip(skip).limit(per_page))

    total_pages = (total + per_page - 1) // per_page


    return render_template(
        "history.html",
        history=data,
        page=page,
        total_pages=total_pages
    )


# Validation functions
def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def allowed_mime(file):
    return file.mimetype.startswith('audio/')


# HF Spaces requires port 7860 and host 0.0.0.0
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)