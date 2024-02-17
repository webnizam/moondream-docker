from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import re
from threading import Thread
from moondream import Moondream, detect_device
from transformers import TextIteratorStreamer, CodeGenTokenizerFast as Tokenizer


app = FastAPI()

device, dtype = detect_device()
model_id = "vikhyatk/moondream1"
tokenizer = Tokenizer.from_pretrained(model_id)
moondream = Moondream.from_pretrained(model_id).to(device=device, dtype=dtype)
moondream.eval()


def answer_question(img, prompt):
    image_embeds = moondream.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    buffer = []

    def run_inference():
        for new_text in moondream.answer_question(
            image_embeds=image_embeds,
            question=prompt,
            tokenizer=tokenizer,
            streamer=streamer,
        ):
            clean_text = re.sub("<$|END$", "", new_text)
            buffer.append(clean_text.strip("<END"))

    thread = Thread(target=run_inference)
    thread.start()
    thread.join()

    return "".join(buffer)


@app.post("/answer-question/")
async def answer_question_endpoint(
    prompt: str = Form(...), file: UploadFile = File(...)
):
    img = Image.open(io.BytesIO(await file.read()))
    response = next(answer_question(img, prompt))
    return JSONResponse({"response": response})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
