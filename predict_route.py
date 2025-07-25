import numpy as np
from flask import Blueprint, request, jsonify
from models_loader import get_models, load_turma_target_names, load_means
from preproc_utils import get_file_as_array, remover_acentos
from features import (
    get_features_histograma,
    get_features_sobel_canny_histograma_gabor_log,
    get_features_original_histograma_gabor
)
# supabase
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime

# criando blueprint para a rota de predição
predict_blueprint = Blueprint('predict', __name__)

# carregando modelos e dados necessários
turma_target_names = load_turma_target_names('dataset/turma_target_names.pkl')
means = load_means({
    'original': 'dataset/means/mean_original.pkl',
    'sobel': 'dataset/means/mean_sobel.pkl',
    'canny': 'dataset/means/mean_canny.pkl',
    'histograma': 'dataset/means/mean_histograma.pkl',
    'lbp': 'dataset/means/mean_lbp.pkl',
    'gabor': 'dataset/means/mean_gabor.pkl',
    'dog': 'dataset/means/mean_dog.pkl',
    'log': 'dataset/means/mean_log.pkl'
})

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_KEY or not SUPABASE_URL:
    raise ValueError("SUPABASE_URL ou SUPABASE_KEY não estão definidos")
else:
    print("Supabase URL e Key carregados com sucesso.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@predict_blueprint.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado..."}), 400

    n_components = request.form.get('n_components', type=float)

    if n_components is None:
        return jsonify({"error": "Parâmetro 'n_components' (inteiro) é obrigatório."}), 400
        
    n_components = int(n_components) if n_components - int(n_components) == 0 else n_components

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nome de arquivo vazio."}), 400

    modelo_svm = 'original_histograma_gabor'
    models, error_message = get_models(n_components, modelo_svm)

    if error_message:
        return jsonify({"error": f"Falha ao carregar os modelos necessários no servidor: {error_message}"}), 500

    try:
        face_1d_normalized = get_file_as_array(file)
        
        if face_1d_normalized is None:
            return jsonify({"error": "Nenhum rosto foi detectado."}), 400

        svm_models_features = {
            'histograma': lambda face, models: get_features_histograma(face, models, means),
            'sobel_canny_histograma_gabor_log': lambda face, models: get_features_sobel_canny_histograma_gabor_log(face, models, means),
            'original_histograma_gabor': get_features_original_histograma_gabor
        }

        final_features = svm_models_features[modelo_svm](face_1d_normalized, models)

        decision_scores = models['svm'].decision_function(final_features)[0]
        exponenciais = np.exp(decision_scores)
        probabilidades = exponenciais / np.sum(exponenciais)
        top5_indices = np.argsort(probabilidades)[::-1][:5]

        top5_results = []
        top5 = []
        top5_chances = []
        for i, idx in enumerate(top5_indices):
            top5_results.append({
                "posicao": f"Top {i+1}",
                "aluno": turma_target_names[idx],
                "confianca": f"{probabilidades[idx]:.2%}"
            })
            top5.append(turma_target_names[idx])
            top5_chances.append(probabilidades[idx])

        # Upload no bucket

        try:
            filename = f"{datetime.utcnow().isoformat()}_{remover_acentos(file.filename)}"
            storage_path = f"thisismebucket/{remover_acentos(top5[0])}/{n_components}/{filename}"

            # resetando o ponteiro do arquivo para o início
            file.seek(0)

            print('Fazendo upload do arquivo para o bucket do Supabase...')
            res = supabase.storage.from_("thisismebucket").upload(path=storage_path, file=file.read(), file_options={ "content-type": file.content_type })
            print('Arquivo enviado para o Supabase com sucesso')
            
            print('Buscando a URL pública do arquivo enviado...')
            public_url = supabase.storage.from_("thisismebucket").get_public_url(storage_path)
            print('URL pública do arquivo obtida com sucesso:', public_url)

            data = {
                "file_url": public_url,
                "top1": top5[0],
                "top2": top5[1],
                "top3": top5[2],
                "top4": top5[3],
                "top5": top5[4],
                "top1_chance": top5_chances[0],
                "top2_chance": top5_chances[1],
                "top3_chance": top5_chances[2],
                "top4_chance": top5_chances[3],
                "top5_chance": top5_chances[4],
                "n_components": n_components,
            }

            print('Salvando inferência na tabela "inferencia"...')
            supabase.table("inferencia").insert(data).execute()
            
        except Exception as e:
            print(f"Não foi possível salvar o arquivo no bucket: {str(e)}")

        return jsonify({"ranking_top_5": top5_results})

    except Exception as e:
        return jsonify({"error": f"Ocorreu um erro inesperado no servidor: {str(e)}"}), 500