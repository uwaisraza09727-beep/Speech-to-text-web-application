# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System dependencies ────────────────────────────────────────────────────────
# ffmpeg is required by openai-whisper for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (HF Spaces requirement) ─────────────────────────────────────
RUN useradd -m -u 1000 appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install CPU-only torch first (avoids downloading huge CUDA wheels) ─────────
RUN pip install --no-cache-dir \
    "torch==2.2.2+cpu" \
    "torchaudio==2.2.2+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python dependencies ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# ── Create uploads directory with correct permissions ─────────────────────────
RUN mkdir -p /app/uploads && chown -R appuser:appuser /app

# ── Switch to non-root user ────────────────────────────────────────────────────
USER appuser

# ── Expose port (HF Spaces uses 7860) ─────────────────────────────────────────
EXPOSE 7860

# ── Start the app with gunicorn ────────────────────────────────────────────────
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "app:app"]
