from wajeezai_api.services.image_processor import ImageProcessor

# images=[r"E:\Projects\WajeezAI\test_images\1.png",r"E:\Projects\WajeezAI\test_images\2.png"]

images = [
    ImageProcessor.load_image_input(r"E:\Projects\WajeezAI\test_images\1.png", timestamp=192.4),
    ImageProcessor.load_image_input(r"E:\Projects\WajeezAI\test_images\2.png"),
]
images_txt = ImageProcessor.run_gemma_batch_google_api(images, subject="منطق ضبابي", lecture_title="التفكير التقريبي")
parsed_output = ImageProcessor.parse_output(images_txt)
parsed_output_with_meta = ImageProcessor.attach_metadata(parsed_output, images)
print('************************************************')
print(parsed_output)
print('************************************************')
print('parsed_output_with_meta')
print(parsed_output_with_meta)

