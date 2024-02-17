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
    thread = Thread(
        target=moondream.answer_question,
        kwargs={
            "image_embeds": image_embeds,
            "question": prompt,
            "tokenizer": tokenizer,
            "streamer": streamer,
        },
    )
    thread.start()

    buffer = ""
    for new_text in streamer:
        clean_text = re.sub("<$|END$", "", new_text)
        buffer += clean_text
        print(buffer)
        yield buffer.strip("<END")


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
