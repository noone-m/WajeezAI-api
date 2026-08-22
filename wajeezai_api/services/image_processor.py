from ast import Dict
import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, List, Union,Dict
from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageOps, ImageEnhance,ImageDraw, ImageFont
from google.genai import types, client
import os

env_path = ".env"

print("env path")
print(env_path)
load_dotenv(dotenv_path=env_path)


client = client.Client(
api_key=os.getenv("GEMINI_API_KEY")
)


@dataclass
class ImageInput:
    image: Image.Image
    path: str
    timestamp: float | None = None 

@dataclass
class Visual:
    coord: list[int]
    desc: str


@dataclass
class SlideResult:
    slide_number: int
    text: str
    visuals: list[Visual]
    image_path: str | None = None
    timestamp: float | None = None

class ImageProcessor:

    @staticmethod
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

    @staticmethod
    def load_image_input(path: str, timestamp: float | None = None, apply_grey: bool = False) -> ImageInput:
        img = ImageProcessor.preprocess_image(path, apply_grey=apply_grey)
        return ImageInput(image=img, path=path, timestamp=timestamp)
    
    @staticmethod
    def run_gemma_batch_google_api(
        images: list["ImageInput"],
        subject: str = "Unknown unfortunately",
        lecture_title: str = "Unknown unfortunately",
    ) -> str:
        for item in images:
            if not isinstance(item.image, Image.Image):
                return f"Images should have the type of PIL.Image.Image not {type(item.image)}"

        prompt = (
            f"SYSTEM: You are an academic transcription expert for {subject}.\n"
            f"LECTURE: {lecture_title}\n"
            "TASK: Analyze the provided images in sequence. For EACH image:\n"
            "1. EXTRACTED TEXT: All text exactly as shown. Support Arabic/English RTL.\n"
            "2. VISUALS: Detect every diagram, sketch, or complex chart. For each, provide:\n"
            "   - BOX: [ymin, xmin, ymax, xmax] (normalized 0-1000 coordinates).\n"
            "   - DESCRIPTION: Formal explanation of the visual's meaning.\n\n"
            "3. For the images in this batch, number them as SLIDE_1, SLIDE_2, SLIDE_3 respectively.\n"
            "OUTPUT FORMAT (STRICT):\n"
            "### SLIDE_[N] ###\n"
            "## TEXT ##\n[Raw Text]\n"
            "## VISUALS ##\n"
            "#\nCOORD: [ymin, xmin, ymax, xmax]\nDESC: [Brief Description Arabic/(English Terms) ]\n#\n"
        )

        try:
            contents = [*(item.image for item in images), prompt]

            response = client.models.generate_content(
                model="gemma-4-31b-it",  # Specific Gemma 4 31B model endpoint
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type='text/plain',
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=False,
                        thinking_level='MINIMAL',)
                )
            )
            print('response', response)
            
            return response.text.strip()

        except Exception as e:
            print(f"Batch inference error: {e}")
            return f"Error: {e}"

    def parse_output(output: str) -> list[SlideResult]:
        """
        Parses the strict VLM output structure (Gemma 4 or Gemini) into clean, 
        easy-to-use Python data.
        
        Input format (exactly as you showed):
        ### SLIDE_1 ###
        ## TEXT ##
        [extracted text here]

        ## VISUALS ##
        # 
        COORD: [ymin, xmin, ymax, xmax]
        DESC: [description]
        #
        COORD: [ymin, xmin, ymax, xmax]
        DESC: [description]
        #
        ### SLIDE_2 ###
        ...

        Returns:
            List[Dict] - one dict per slide
            [
                {
                    "slide_number": 1,
                    "text": "full raw text with Arabic/English...",
                    "visuals": [
                        {"coord": [90, 85, 210, 365], "desc": "Ring topology diagram..."},
                        {"coord": [320, 0, 400, 185], "desc": "Mesh formula box..."}
                    ]
                },
                ...
            ]
        """
    
        slides: list[SlideResult] = []
        
        slide_blocks = re.split(
            r'### SLIDE_(\d+) ###',
            output.strip()
        )

        for i in range(1, len(slide_blocks), 2):
            slide_num = int(slide_blocks[i])
            content = slide_blocks[i + 1]
            if "## VISUALS ##" in content:
                text_part, visuals_part = content.split(
                    "## VISUALS ##",
                    1
                )

            else:
                text_part = content
                visuals_part = ""

            text = text_part.replace("## TEXT ##", "").strip()
            visuals: list[Visual] = []

            pattern = (
                r'COORD:\s*(\[.*?\])'
                r'\s*DESC:\s*(.*?)(?=\s*#|\Z)'
            )

            matches = re.findall(
                pattern,
                visuals_part,
                re.DOTALL | re.MULTILINE
            )

            for coord_str, desc in matches:
                try:
                    coord = ast.literal_eval(coord_str.strip())
                    visuals.append(
                        Visual(
                            coord=coord,
                            desc=desc.strip()
                        )
                    )
                except (ValueError, SyntaxError):
                    continue

            slides.append(
                SlideResult(
                    slide_number=slide_num,
                    text=text,
                    visuals=visuals
                )
            )
        return slides

    @staticmethod
    def attach_metadata(
        slides: list[SlideResult],
        images: list[ImageInput]
    ) -> list[SlideResult]:

        if len(slides) != len(images):
            print(
                f"Warning: slide count ({len(slides)}) != "
                f"image count ({len(images)}); "
                f"metadata attachment may be misaligned."
            )

        for slide, item in zip(slides, images):
            slide.image_path = item.path
            slide.timestamp = item.timestamp

        return slides

    
    ################## Drawing Boxes Logic ##################

    def _get_image_coord(parsed_output: List[Dict[str, Any]],image_path) -> dict:
        coordinates = {}
        try:
            for visual in parsed_output['visuals']:
                coordinates[image_path] = visual['coord']
                return coordinates
        except Exception as e:
            print(f"Error retrieving image{image_path} coordinates: {e}")
            return {}  

    def _get_images_coord(parsed_output: List[Dict[str, Any]],image_paths: List[str]) -> list[dict]:
        return [ImageProcessor._get_image_coord(parsed_output, image_path) for image_path in image_paths]

    def _draw_boxes_on_image(
        image: Union[str, Image.Image],
        boxes: List[List[int]],
        width: int = 3,
        with_labels: bool = False
    ):
        """
        Draw boxes where coords are:
        [ymin, xmin, ymax, xmax] normalized in range 0–1000
        """

        # Load image if path
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        W, H = img.size
        draw = ImageDraw.Draw(img)
        
        def get_color(i, total):
            import colorsys
            hue = i / total
            rgb = colorsys.hsv_to_rgb(hue, 1, 1)
            return tuple(int(c * 255) for c in rgb)
        
        for i, box in enumerate(boxes):
            ymin, xmin, ymax, xmax = box

            # Convert normalized → pixel coordinates
            x1 = int((xmin / 1000) * W)
            y1 = int((ymin / 1000) * H)
            x2 = int((xmax / 1000) * W)
            y2 = int((ymax / 1000) * H)

            color = get_color(i, len(boxes))

            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

            if with_labels:
                draw.text((x1, max(0, y1 - 12)), f"Box {i+1}", fill=color)

        return img


    ###############################    
    