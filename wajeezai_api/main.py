import json
import os
import time

from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


UPLOAD_DIR = "uploads"


@app.post("/api/lectures/upload")
async def upload_lecture(
    audio: UploadFile = File(...),
    slides: list[UploadFile] = File(default=[]),
    timestamps: UploadFile = File(...),
):
    
    lecture_id = "lecture_" + str(int(time.time()))
    lecture_dir = os.path.join(UPLOAD_DIR, lecture_id)
    os.makedirs(lecture_dir, exist_ok=True)

    # Save audio
    audio_path = os.path.join(lecture_dir, "lecture.wav")

    with open(audio_path, "wb") as f:
        while chunk := await audio.read(1024 * 1024):
            f.write(chunk)

    # Save slides
    for slide in slides:
        slide_path = os.path.join(
            lecture_dir,
            os.path.basename(slide.filename)
        )

        with open(slide_path, "wb") as f:
            while chunk := await slide.read(1024 * 1024):
                f.write(chunk)

    # Read and save timestamps
    try:
        timestamps_content = await timestamps.read()
        timestamps_data = json.loads(timestamps_content)

        timestamps_path = os.path.join(
            lecture_dir,
            "timestamps.json"
        )

        with open(timestamps_path, "w", encoding="utf-8") as f:
            json.dump(
                timestamps_data,
                f,
                ensure_ascii=False,
                indent=2
            )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid timestamps JSON"
        )

    return {
        "status": "ok",
        "lecture_id": lecture_id,
        "slides": len(timestamps_data),
    }

@app.post("/api/lectures/{lecture_id}/process")
async def process_lecture(lecture_id: str):
    lecture_dir = os.path.join(UPLOAD_DIR, lecture_id)
    if not os.path.isdir(lecture_dir):
        raise HTTPException(404, "Lecture not found")

    # run your pipeline here: AudioProcessor -> ImageProcessor -> Alignment.align
    # save result as lecture_dir/result.json

    return {"status": "processing_complete", "lecture_id": lecture_id}