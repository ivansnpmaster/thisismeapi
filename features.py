import numpy as np
from preproc_filters import (
    filtro_sobel_1d,
    filtro_canny_1d,
    filtro_histogram_equalization_1d,
    filtro_gabor_1d,
    filtro_log_1d
)

def get_features_histograma(face_1d_normalized, models, means):
    histogram_1d = filtro_histogram_equalization_1d(face_1d_normalized)
    return models['pca_histograma'].transform([histogram_1d - means['histograma']])

def get_features_sobel_canny_histograma_gabor_log(face_1d_normalized, models, means):
    sobel_1d = filtro_sobel_1d(face_1d_normalized)
    canny_1d = filtro_canny_1d(face_1d_normalized)
    histogram_1d = filtro_histogram_equalization_1d(face_1d_normalized)
    gabor_1d = filtro_gabor_1d(face_1d_normalized)
    log_1d = filtro_log_1d(face_1d_normalized)

    feat_sobel = models['pca_sobel'].transform([sobel_1d - means['sobel']])
    feat_canny = models['pca_canny'].transform([canny_1d - means['canny']])
    feat_histograma = models['pca_histograma'].transform([histogram_1d - means['histograma']])
    feat_gabor = models['pca_gabor'].transform([gabor_1d - means['gabor']])
    feat_log = models['pca_log'].transform([log_1d - means['log']])

    return np.hstack([feat_sobel, feat_canny, feat_histograma, feat_gabor, feat_log])

def get_features_original_histograma_gabor(face_1d_normalized, models):
    histogram_1d = filtro_histogram_equalization_1d(face_1d_normalized)
    gabor_1d = filtro_gabor_1d(face_1d_normalized)

    feat_original = models['pca_original'].transform([face_1d_normalized])
    feat_histograma = models['pca_histograma'].transform([histogram_1d])
    feat_gabor = models['pca_gabor'].transform([gabor_1d])

    return np.hstack([feat_original, feat_histograma, feat_gabor])