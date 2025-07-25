import numpy as np
import cv2
from skimage.feature import local_binary_pattern

def filtro_sobel_1d(img_1d):
    img_2d = img_1d.reshape((125,94))
    sobelx = cv2.Sobel(img_2d, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_2d, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8).flatten() / 255.0

def filtro_canny_1d(img_1d):
    img_2d = img_1d.reshape((125,94))
    if img_2d.dtype != np.uint8:
        img_2d = (img_2d*255).astype(np.uint8)
    edges = cv2.Canny(img_2d, 50, 150)
    return edges.flatten() / 255.0

def filtro_histogram_equalization_1d(img_1d):
    """Equaliza histograma e retorna vetor 1D"""
    img_2d = img_1d.reshape((125,94))
    if img_2d.dtype != np.uint8:
        img_2d = (img_2d * 255).astype(np.uint8)
    eq_2d = cv2.equalizeHist(img_2d)
    return eq_2d.flatten() / 255.0

def filtro_lbp_1d(img_1d):
    """Calcula o Local Binary Pattern, que é robusto à iluminação."""
    img_2d = (img_1d.reshape((125, 94)) * 255).astype(np.uint8)
    # P: número de pontos vizinhos. R: raio do círculo.
    # 'uniform' é o método mais comum e eficaz.
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img_2d, n_points, radius, method='uniform')
    # Normaliza o resultado para o intervalo 0-1 para consistência.
    lbp = (lbp - np.min(lbp)) / (np.max(lbp) - np.min(lbp))
    return lbp.flatten()

def filtro_gabor_1d(img_1d):
    """
    Aplica um filtro Gabor e retorna a magnitude da resposta, que é robusta e visualmente rica.
    """
    # Converte o vetor 1D para uma imagem 2D no formato uint8 (0-255)
    img_2d = (img_1d.reshape((125, 94)) * 255).astype(np.uint8)

    # Parâmetros do filtro Gabor ajustados para características faciais
    ksize = (11, 11)   # Kernel um pouco maior para capturar texturas mais amplas
    sigma = 4.0        # Desvio padrão da gaussiana
    theta = np.pi / 4  # Orientação de 45 graus
    lambd = 10.0       # Comprimento de onda da parte senoidal
    gamma = 0.5        # Proporção do aspecto elíptico

    # Cria o kernel Gabor
    gabor_kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)

    # Aplica o filtro. O resultado pode ter valores negativos.
    gabor_response = cv2.filter2D(img_2d, cv2.CV_32F, gabor_kernel)

    # O PASSO CRUCIAL: Usamos a magnitude (valor absoluto) da resposta do filtro.
    # Isso representa a "energia" da textura naquela orientação, e é sempre positivo.
    magnitude = np.abs(gabor_response)

    # Normaliza a magnitude para o intervalo 0-1 para consistência com os outros filtros.
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Achata a imagem de volta para um vetor 1D.
    return magnitude_normalized.flatten()

def filtro_dog_1d(img_1d):
    """Aplica o filtro Difference of Gaussians (DoG) para realce de bordas e normalização de iluminação."""
    img_2d = (img_1d.reshape((125, 94)) * 255).astype(np.uint8)

    # Aplica dois filtros gaussianos com desvios padrão diferentes
    blur1 = cv2.GaussianBlur(img_2d, (3, 3), 0)
    blur2 = cv2.GaussianBlur(img_2d, (9, 9), 0)

    # Subtrai um do outro para obter o DoG
    dog_img = blur1 - blur2

    return dog_img.flatten() / 255.0

def filtro_log_1d(img_1d):
    """Aplica o filtro Laplacian of Gaussian (LoG) para detecção de bordas e cantos."""
    img_2d = (img_1d.reshape((125, 94)) * 255).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img_2d, (3, 3), 0)
    log_img = cv2.Laplacian(img_blur, cv2.CV_64F)
    log_img = cv2.convertScaleAbs(log_img)
    return log_img.flatten() / 255.0