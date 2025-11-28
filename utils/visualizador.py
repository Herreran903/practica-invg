import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def visualizar_npy(ruta_archivo):
    """Carga y visualiza un archivo .npy."""

    if not os.path.exists(ruta_archivo):
        print(f"❌ Error: Archivo no encontrado en la ruta: {ruta_archivo}")
        sys.exit(1)

    try:
        data = np.load(ruta_archivo)
        print(
            f"✅ Archivo cargado exitosamente. Forma de los datos (Shape): {data.shape}"
        )

    except Exception as e:
        print(f"❌ Error al cargar el archivo .npy: {e}")
        sys.exit(1)

    dim = data.ndim

    plt.figure()

    if dim == 2:
        plt.imshow(data, cmap="gray")
        plt.title(f"Visualización 2D (Matriz {data.shape[0]}x{data.shape[1]})")
        plt.colorbar(label="Valores Normalizados")
        plt.xlabel("Eje X (Columna)")
        plt.ylabel("Eje Y (Fila)")

    elif dim == 1:
        plt.plot(data)
        plt.title(f"Visualización 1D (Vector de tamaño {data.shape[0]})")
        plt.xlabel("Índice de Tiempo/Muestra")
        plt.ylabel("Valor")

    elif dim == 3 and data.shape[-1] in [3, 4]:
        plt.imshow(data)
        plt.title(f"Visualización 3D (Imagen de color {data.shape})")

    else:
        print(
            f"⚠️ Aviso: No se puede visualizar automáticamente un array de {dim} dimensiones con esta forma."
        )
        print("Muestra la forma manualmente o ajusta el script.")
        return

    plt.show()


if __name__ == "__main__":
    archivo_a_visualizar = "ruta_a_tu_archivo.npy"

    if len(sys.argv) > 1:
        archivo_a_visualizar = sys.argv[1]

    visualizar_npy(archivo_a_visualizar)
