from pydantic import BaseModel

class ProcessInput(BaseModel):
    sentence: str
    returnPasal: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sentence": "Tono, menggugat pemilik apartemen, PT Sejahtera Properti, atas pelanggaran kontrak sewa yang menyebabkan kerugian padanya. Tono telah menyewa apartemen tersebut selama dua tahun dengan kontrak yang menetapkan bahwa pemilik apartemen bertanggung jawab untuk memperbaiki kerusakan yang disebabkan oleh keausan normal.Namun, setelah beberapa bulan tinggal di apartemen, sistem pipa air apartemen mengalami kebocoran parah yang menyebabkan banjir di dalam apartemen dan merusak perabotan serta barang-barang pribadinya. Tono segera melaporkan kejadian tersebut kepada PT Sejahtera Properti dan meminta perbaikan segera. Namun, PT Sejahtera Properti gagal menanggapi keluhan Tono dalam waktu yang wajar dan tidak memperbaiki kerusakan tersebut.",
                    "returnPasal": True,
                }
            ]
        }
    }
