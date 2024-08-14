from fastapi import FastAPI, status

from app.process_input import ProcessInput
# from app.langkah_hukum.process_input_gemini import ProcessInputGemini

from app.Model import IndoSbertModel
# from app.langkah_hukum.gemini import generate_response

app = FastAPI()

indosbert = IndoSbertModel()

@app.get("/")
async def root():
    return {"message": "Hello Banjarmasin"}

@app.post("/get-pasal", status_code=status.HTTP_201_CREATED)
def getPasal(input: ProcessInput):
    sentence = input.sentence
    return_pasal = input.returnPasal

    if return_pasal:
        output = indosbert.getDatabaseVector(sentence, returnPredict=return_pasal)

        return {"data": output}
    else:
        return "Return Pasal is False"

# @app.post("/langkah-hukum", status_code=status.HTTP_201_CREATED)
# def getLangkahHukum(input: ProcessInputGemini):
#     sentence = input.sentence

#     try:
#         output = generate_response(sentence)
#         return output
        
#     except Exception as e:
#         return {"error": str(e)}