from paddleocr import PaddleOCR
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import json
import gradio as gr

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_pdserving=False, cls_batch_num=8, det_batch_num=8, rec_batch_num=8)

def index(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    resize_factor = 1
    new_size = tuple(int(dim * resize_factor) for dim in img.size)
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    img_array = np.array(img.convert('RGB'))

    result = ocr.ocr(img_array)

    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    print(boxes)
    print(txts)

    output_dict = {"texts": txts, "boxes": boxes, "scores": scores}
    output_json = json.dumps(output_dict) 

    return output_json


inputs_image_url = [
    gr.Textbox(type="text", label="Image URL"),
]

outputs_result_json = [
    gr.Textbox(type="text", label="Result JSON"),
]

interface_image_url = gr.Interface(
    fn=index,
    inputs=inputs_image_url,
    outputs=outputs_result_json,
    title="Text Extraction",
    cache_examples=False,
).queue().launch()
