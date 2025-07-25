import numpy as np
import cv2

haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_rate(img, x, y, largura, altura, largura_lfw=94, altura_lfw=125, interpolation=cv2.INTER_AREA):
    razao_aspecto = altura_lfw / largura_lfw
    centro_x = x + largura / 2
    centro_y = y + altura / 2
    area = largura * altura
    largura_adj = np.sqrt(area / razao_aspecto)
    altura_adj = razao_aspecto * largura_adj
    x_min = int(np.floor(centro_x - largura_adj / 2))
    x_max = int(np.ceil(centro_x + largura_adj / 2))
    y_min = int(np.floor(centro_y - altura_adj / 2 + 0.5))
    y_max = int(np.ceil(centro_y + altura_adj / 2 + 0.5))
    if y_min < 0: y_max -= y_min; y_min = 0
    if x_min < 0: x_max -= x_min; x_min = 0
    crop_img = img[y_min:y_max, x_min:x_max]
    img_lfw = cv2.resize(crop_img, (largura_lfw, altura_lfw), interpolation=interpolation)
    return img_lfw.flatten()

def get_file_as_array(file):
    in_memory_file = file.read()
    img_array = np.frombuffer(in_memory_file, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = haar_face_cascade.detectMultiScale(img_gray)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    face_processed = crop_rate(img_gray, x, y, w, h)
    face_1d_normalized = face_processed / 255.0
    return face_1d_normalized

def remover_acentos(texto):
    substituicoes = {
        'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c',
        'Á': 'A', 'À': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
        'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
        'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
        'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
        'Ç': 'C'
    }
    return ''.join(substituicoes.get(c, c) for c in texto)