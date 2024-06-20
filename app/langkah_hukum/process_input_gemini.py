from pydantic import BaseModel

class ProcessInputGemini(BaseModel):
    sentence: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sentence": "saya digugat cerai istri saya karena saya tidak bisa menafkahi istri dan anak saya dan mereka meminta haknya. Namun saya tidak dapat memenuhi seluruh hak yang mereka minta karena saya tidak memiliki pekerjaan tetap. Apa yang harus saya lakukan?",
                }
            ]
        }
    }
