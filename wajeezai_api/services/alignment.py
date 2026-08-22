from dataclasses import dataclass
from typing import Optional,Union
import re
import numpy as np
from wajeezai_api.services.audio_processor import TranscriptionSegment, TranscriptionResult
from wajeezai_api.services.image_processor import SlideResult, Visual
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os 
import math

from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# Final result structures
# ============================================================

@dataclass
class TextFusion:
    ocr_chunk: str
    asr_chunks: list[TranscriptionSegment]


@dataclass
class VisualFusion:
    path: str | None
    desc: dict


@dataclass
class SlideFusion:
    slide_number: int
    texts: list[TextFusion]
    visuals: list[VisualFusion]
    
    
class Alignment:
    @staticmethod
    def align(
        slides: list[SlideResult], # output of image_processor pipeline
        transcription: TranscriptionResult, # output of audio_processor pipeline
        text_threshold: float = 0.50,
        visual_threshold: float = 0.55,
        text_top_k: int = 3,
        visual_top_k: int = 3,
    ) -> list[SlideFusion]:

        slides_list = []

        for slide in slides:

            result = Alignment.process_slide(
                slide=slide,
                transcription=transcription,
                text_threshold=text_threshold,
                visual_threshold=visual_threshold,
                text_top_k=text_top_k,
                visual_top_k=visual_top_k
            )

            slides_list.append(result)

        return slides_list
        
    alpha: float = 0.7,
    beta: float = 0.3

    
    # ============================================================
    # temporal_similarity
    # ============================================================

    def temporal_similarity(
        image_timestamp: float,
        segment: TranscriptionSegment,
        sigma: float = 10.0
    ) -> float:
        """
        Calculate temporal similarity between a slide timestamp
        and an ASR segment.

        The ASR segment timestamp is represented by its midpoint.

        S_temporal = exp(-(t_image - t_chunk)^2 / (2 * sigma^2))
        """

        if image_timestamp is None:
            return 0.0

        # Middle of ASR segment
        t_chunk = (segment.start + segment.end) / 2

        # Time difference
        difference = image_timestamp - t_chunk

        # Gaussian temporal similarity
        score = math.exp(
            -(difference ** 2) / (2 * sigma ** 2)
        )

        return score
    # ============================================================
    # Crop Illustrations
    # ============================================================
    def preprocess_image( image: Union[str, Path, np.ndarray, Image.Image], apply_grey: bool) -> Image.Image:
        """Light preprocessing (contrast boost) to improve OCR accuracy on both engines."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # resize first (important)
        img = img.resize((int(img.width * 1.5), int(img.height * 1.5)), Image.BICUBIC)

        if apply_grey:
            # convert to grayscale
            img = img.convert("L")
    
        # Boost contrast – helps especially with Arabic script and low-quality scans
        img = ImageOps.exif_transpose(img) # Reads EXIF tag Orientation
        enhancer = ImageEnhance.Contrast(img) 
        return enhancer.enhance(2)

    def crop_illustrations(visual, image_path, idx=0, output_dir="cropped_images"):
        os.makedirs(output_dir, exist_ok=True)  # also fixes the earlier missing-dir issue

        img = Alignment.preprocess_image(image_path, apply_grey=False)
        W, H = img.size

        ymin, xmin, ymax, xmax = visual.coord
        x1 = int((xmin / 1000) * W)
        y1 = int((ymin / 1000) * H)
        x2 = int((xmax / 1000) * W)
        y2 = int((ymax / 1000) * H)

        if x2 <= x1 or y2 <= y1:
            return None  # degenerate box — nothing to crop

        crop_img = img.crop((x1, y1, x2, y2))
        base_name = os.path.basename(image_path).split('.')[0]
        crop_filename = f"{base_name}_crop_{idx}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        crop_img.save(crop_path)

        return crop_path
    # ============================================================
    # Text utilities
    # ============================================================

    def clean_text(text: str) -> str:

        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def chunk_text(
        text: str,
        chunk_size: int = 300,
        overlap: int = 50
    ) -> list[str]:

        text = Alignment.clean_text(text)

        if not text:
            return []

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        chunks = []

        step = chunk_size - overlap

        for start in range(0, len(text), step):

            chunk = text[
                start:start + chunk_size
            ].strip()

            if chunk:
                chunks.append(chunk)

            if start + chunk_size >= len(text):
                break

        return chunks


    # ============================================================
    # Embeddings
    # ============================================================

    def get_embeddings(
        texts: list[str],
        batch_size: int = 32
    ) -> np.ndarray | None:

        texts = [
            Alignment.clean_text(text)
            for text in texts
            if Alignment.clean_text(text)
        ]

        if not texts:
            return None

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.asarray(embeddings)


    # ============================================================
    # Match OCR chunks with ASR
    # ============================================================

    def match_slide_text(
        slide_chunks: list[str],
        transcription_segments: list[TranscriptionSegment],
        threshold: float = 0.50,
        top_k_per_chunk: int = 3,
        timestamp:float=None
    ) -> list[TextFusion]:

        if not slide_chunks:
            return []

        if not transcription_segments:
            return []

        # --------------------------------------------------------
        # Encode OCR chunks
        # --------------------------------------------------------

        slide_embeddings = Alignment.get_embeddings(
            slide_chunks
        )

        # --------------------------------------------------------
        # Encode ASR segments
        # --------------------------------------------------------

        asr_texts = [
            segment.text
            for segment in transcription_segments
        ]

        asr_embeddings = Alignment.get_embeddings(
            asr_texts
        )

        if (
            slide_embeddings is None
            or asr_embeddings is None
        ):
            return []

        # --------------------------------------------------------
        # Similarity
        # --------------------------------------------------------

        similarity_matrix = cosine_similarity(
            slide_embeddings,
            asr_embeddings
        )


        temporal_scores = np.array([
            Alignment.temporal_similarity(timestamp, segment) for segment in transcription_segments
        ])


        results = []

        # --------------------------------------------------------
        # For every OCR chunk
        # --------------------------------------------------------

        for slide_idx, slide_chunk in enumerate(
            slide_chunks
        ):

            semantic_scores = similarity_matrix[slide_idx]

            final_scores = Alignment.alpha * semantic_scores + Alignment.beta * temporal_scores
            ranked_indices = np.argsort(final_scores)[::-1]


            matched_asr = []

            for asr_idx in ranked_indices:

                score = float(final_scores[asr_idx])

                # Since sorted descending, once we are below
                # threshold, remaining scores will also be below it.
                if score < threshold:
                    break

                matched_asr.append(
                    transcription_segments[asr_idx]
                )

                # if len(matched_asr) >= top_k_per_chunk:
                #     break

            # ----------------------------------------------------
            # Add this OCR chunk
            # ----------------------------------------------------

            results.append(
                TextFusion(
                    ocr_chunk=slide_chunk,
                    asr_chunks=matched_asr
                )
            )

        return results


    # ============================================================
    # Match visual descriptions with ASR
    # ============================================================

    def match_visuals(
        visuals: list[Visual],
        transcription_segments: list[TranscriptionSegment],
        threshold: float = 0.45,
        top_k_per_visual: int = 3,
        timestamp:float|None=None,
        image_path: str | None = None,
    ) -> list[VisualFusion]:

        if not visuals:
            return []

        if not transcription_segments:
            return []

        # --------------------------------------------------------
        # Only visuals with descriptions
        # --------------------------------------------------------

        valid_visuals = [
            visual
            for visual in visuals
            if Alignment.clean_text(visual.desc)
        ]

        if not valid_visuals:
            return []

        # --------------------------------------------------------
        # Encode visual descriptions
        # --------------------------------------------------------

        visual_texts = [
            visual.desc
            for visual in valid_visuals
        ]

        visual_embeddings = Alignment.get_embeddings(
            visual_texts
        )

        # --------------------------------------------------------
        # Encode ASR
        # --------------------------------------------------------

        asr_texts = [
            segment.text
            for segment in transcription_segments
        ]

        asr_embeddings = Alignment.get_embeddings(
            asr_texts
        )

        if (
            visual_embeddings is None
            or asr_embeddings is None
        ):
            return []

        # --------------------------------------------------------
        # Similarity
        # --------------------------------------------------------

        similarity_matrix = cosine_similarity(
            visual_embeddings,
            asr_embeddings
        )

        temporal_scores = np.array([
            Alignment.temporal_similarity(timestamp, segment) for segment in transcription_segments
        ])

        results = []

        # --------------------------------------------------------
        # For every visual
        # --------------------------------------------------------

        for visual_idx, visual in enumerate(
            valid_visuals
        ):
            semantic_scores = similarity_matrix[visual_idx]
            final_scores = Alignment.alpha * semantic_scores + Alignment.beta * temporal_scores
            ranked_indices = np.argsort(final_scores)[::-1]

            matched_asr = []

            for asr_idx in ranked_indices:

                score = float(final_scores[asr_idx])

                if score < threshold:
                    break

                matched_asr.append(
                    transcription_segments[asr_idx]
                )

                # if len(matched_asr) >= top_k_per_visual:
                #     break

            # ----------------------------------------------------
            # Add visual
            # ----------------------------------------------------
            visual_path = Alignment.crop_illustrations(visual,image_path,visual_idx)
            results.append(
                VisualFusion(
                    path=visual_path,
                    desc={
                        "ocr_chunk": visual.desc,
                        "asr_chunks": matched_asr
                    }
                )
            )

        return results


    # ============================================================
    # Process ONE slide
    # ============================================================

    def process_slide(
        slide: SlideResult,
        transcription: TranscriptionResult,
        text_threshold: float = 0.50,
        visual_threshold: float = 0.45,
        text_top_k: int = 3,
        visual_top_k: int = 3,
    ) -> SlideFusion:

        # --------------------------------------------------------
        # 1. Chunk OCR text
        # --------------------------------------------------------

        slide_chunks = Alignment.chunk_text(
            slide.text,
            chunk_size=300,
            overlap=50
        )

        # --------------------------------------------------------
        # 2. Match OCR chunks with ASR
        # --------------------------------------------------------

        text_results = Alignment.match_slide_text(
            slide_chunks=slide_chunks,
            transcription_segments=transcription.segments,
            threshold=text_threshold,
            timestamp=slide.timestamp,
            top_k_per_chunk=text_top_k
        )

        # --------------------------------------------------------
        # 3. Match visuals with ASR
        # --------------------------------------------------------

        visual_results = Alignment.match_visuals(
            visuals=slide.visuals,
            transcription_segments=transcription.segments,
            threshold=visual_threshold,
            top_k_per_visual=visual_top_k,
            timestamp=slide.timestamp,
            image_path=slide.image_path
        )

        # --------------------------------------------------------
        # 4. Final slide result
        # --------------------------------------------------------

        return SlideFusion(
            slide_number=slide.slide_number,
            texts=text_results,
            visuals=visual_results
        )

