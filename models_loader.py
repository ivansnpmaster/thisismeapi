import pickle as pk
import os

model_cache = {}

def load_turma_target_names(path):
    with open(path, 'rb') as f:
        return pk.load(f)

def load_means(means_path_dict):
    means = {}
    for key, path in means_path_dict.items():
        with open(path, 'rb') as f:
            means[key] = pk.load(f)
        print(f">>> Média '{key}' carregada com sucesso. <<<")
    return means

def get_models(n_components, pasta_svm):
    if n_components in model_cache:
        print(f">>> Usando modelos em cache para N_COMPONENTS={n_components}. <<<")
        return model_cache[n_components], None

    print(f">>> Cache miss! Carregando modelos do disco para N_COMPONENTS={n_components}... <<<")

    files_to_load = [
        ('pca_original',  'modelos/PCA', 'original'),
        ('pca_histograma','modelos/PCA', 'histograma'),
        ('pca_gabor',     'modelos/PCA', 'gabor'),
        ('svm',           f'modelos/SVM/{pasta_svm}', str(n_components))
    ]

    models = {}
    current_loading_file = None
    try:
        for key, base_path, file_prefix in files_to_load:
            file_path = f'{base_path}/{file_prefix}.pkl' if key == 'svm' else f'{base_path}/{file_prefix}_{n_components}.pkl'
            current_loading_file = file_path
            with open(file_path, 'rb') as f:
                models[key] = pk.load(f)

        model_cache[n_components] = models
        print(f">>> Modelos para N_COMPONENTS={n_components} carregados e armazenados em cache. <<<")
        return models, None

    except FileNotFoundError:
        error_message = f"Arquivo de modelo não encontrado: '{current_loading_file}'"
        print(f"ERRO: {error_message}")
        return None, error_message

    except Exception as e:
        error_message = f"Erro ao carregar o arquivo '{current_loading_file}': {str(e)}"
        print(f"ERRO: {error_message}")
        return None, error_message