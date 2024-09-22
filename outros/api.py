import face_recognition
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Permitir apenas arquivos com estas extensões
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Classe de resposta para o JSON
class FaceRecognitionResponse(BaseModel):
    face_found_in_image: bool
    is_picture_of_obama: bool


# Verifica se o arquivo possui uma extensão válida
def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
    <head>
        <title>Is this a picture of Obama?</title>
    </head>
    <body>
        <h1>Upload a picture and see if it's a picture of Obama!</h1>
        <form action="/uploadfile/" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit">
        </form>
    </body>
    </html>
    """
    return content


@app.post("/uploadfile/", response_model=FaceRecognitionResponse)
async def upload_file(file: UploadFile = File(...)):
    # Verifica se o arquivo tem uma extensão permitida
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format")

    # Carregar a imagem enviada
    image = face_recognition.load_image_file(file.file)

    # Pre-calculado face encoding de Obama
    known_face_encoding = [-0.09634063, 0.12095481, -0.00436332, -0.07643753, 0.0080383,
                           0.01902981, -0.07184699, -0.09383309, 0.18518871, -0.09588896,
                           0.23951106, 0.0986533, -0.22114635, -0.1363683, 0.04405268,
                           0.11574756, -0.19899382, -0.09597053, -0.11969153, -0.12277931,
                           0.03416885, -0.00267565, 0.09203379, 0.04713435, -0.12731361,
                           -0.35371891, -0.0503444, -0.17841317, -0.00310897, -0.09844551,
                           -0.06910533, -0.00503746, -0.18466514, -0.09851682, 0.02903969,
                           -0.02174894, 0.02261871, 0.0032102, 0.20312519, 0.02999607,
                           -0.11646006, 0.09432904, 0.02774341, 0.22102901, 0.26725179,
                           0.06896867, -0.00490024, -0.09441824, 0.11115381, -0.22592428,
                           0.06230862, 0.16559327, 0.06232892, 0.03458837, 0.09459756,
                           -0.18777156, 0.00654241, 0.08582542, -0.13578284, 0.0150229,
                           0.00670836, -0.08195844, -0.04346499, 0.03347827, 0.20310158,
                           0.09987706, -0.12370517, -0.06683611, 0.12704916, -0.02160804,
                           0.00984683, 0.00766284, -0.18980607, -0.19641446, -0.22800779,
                           0.09010898, 0.39178532, 0.18818057, -0.20875394, 0.03097027,
                           -0.21300618, 0.02532415, 0.07938635, 0.01000703, -0.07719778,
                           -0.12651891, -0.04318593, 0.06219772, 0.09163868, 0.05039065,
                           -0.04922386, 0.21839413, -0.02394437, 0.06173781, 0.0292527,
                           0.06160797, -0.15553983, -0.02440624, -0.17509389, -0.0630486,
                           0.01428208, -0.03637431, 0.03971229, 0.13983178, -0.23006812,
                           0.04999552, 0.0108454, -0.03970895, 0.02501768, 0.08157793,
                           -0.03224047, -0.04502571, 0.0556995, -0.24374914, 0.25514284,
                           0.24795187, 0.04060191, 0.17597422, 0.07966681, 0.01920104,
                           -0.01194376, -0.02300822, -0.17204897, -0.0596558, 0.05307484,
                           0.07417042, 0.07126575, 0.00209804]

    # Obter os encodings das faces na imagem enviada
    unknown_face_encodings = face_recognition.face_encodings(image)

    face_found = False
    is_obama = False

    if len(unknown_face_encodings) > 0:
        face_found = True
        # Comparar a primeira face encontrada na imagem com a face de Obama
        match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encodings[0])
        is_obama = match_results[0]

    # Retornar o resultado em JSON
    return FaceRecognitionResponse(face_found_in_image=face_found, is_picture_of_obama=is_obama)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
