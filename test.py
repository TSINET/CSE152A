import numpy as np
import cv2


def photometric_stereo(images, light_directions):
    """
    Calcule les normales de surface et les albédos d'un objet à partir de plusieurs images.

    Parameters
    ----------
    images : np.ndarray
        Tableau contenant les images de l'objet prises avec différentes sources de lumière.
    light_directions : list
        Liste des directions des sources de lumière pour chaque image. Chaque direction est un vecteur avec 3 composantes (x, y, z).

    Returns
    -------
    surface_normals : np.ndarray
        Normales de surface estimées.
    albedos : np.ndarray
        Albédos estimés.
    visualized_normals : np.ndarray
        Normales de surface visualisées en format RGB.
    """
    # Convertir les directions des sources de lumière en une matrice
    L = np.array(light_directions)
    Lt = L.T

    # Obtenir les dimensions de l'image
    height, width = images[0].shape[:2]

    # Initialiser les tableaux pour les normales et les albédos
    surface_normals = np.zeros((height, width, 3))
    albedos = np.zeros((height, width, 3))

    I = np.zeros((len(light_directions), 3))

    # Parcourir chaque pixel de l'image
    for x in range(width):
        for y in range(height):
            for i in range(len(images)):
                I[i] = images[i][y][x]  # Obtenir les intensités pour les 3 images

            # Calculer l'inverse de la transposée des directions de lumière
            tmp1 = np.linalg.inv(Lt)

            # Calculer les normales de surface
            N = np.dot(tmp1, I).T
            rho = np.linalg.norm(N, axis=1)
            albedos[y][x] = rho

            # Calculer la composante de luminosité (gris) de la normale
            N_gray = N[0] * 0.0722 + N[1] * 0.7152 + N[2] * 0.2126

            # Normaliser la normale
            N_norm = np.linalg.norm(N)

            if N_norm == 0:
                continue

            surface_normals[y][x] = N_gray / N_norm

    # Convertir les normales en format RGB
    visualized_normals = ((surface_normals * 0.5 + 0.5) * 255).astype(np.uint8)
    visualized_normals = cv2.cvtColor(visualized_normals, cv2.COLOR_BGR2RGB)

    # Normaliser les albédos
    albedos = (albedos / np.max(albedos) * 255).astype(np.uint8)

    return surface_normals, albedos, visualized_normals
