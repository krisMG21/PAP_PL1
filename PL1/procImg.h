#include "cuda_runtime.h"
#ifndef PROCIMG_H
#define PROCIMG_H

#ifdef __cplusplus
extern "C" {
#endif

    // Estructura para representar un píxel con canales RGB
    typedef struct {
        unsigned char r, g, b;
    } Pixel;

    /**
     * @brief Procesa la imagen usando CUDA segun la opcion dada.
     *
     * @param pixels    [in/out] Puntero al array lineal de píxeles (host).
     *                  Tras la ejecución, contendrá el resultado de la operación.
     * @param height    [in]  Altura de la imagen (número de filas).
     * @param width     [in]  Anchura de la imagen (número de columnas).
     * @param option    [in]  Opción de procesamiento.
     *                     - 1: Conversión a B/N
     *                     - 6: Invertir colores
     *                     - 21: Pixelar en color
     *                     - 22: Pixelar en B/N
     *                     - 31: Identificar color rojo
     *                     - 32: Identificar color verde
     *                     - 33: Identificar color azul
     * @param filterDiv [in]  Factor para redimensionar el blockDim (ej.: si pixelar).
     * @param outCount  [out] Puntero a variable donde se guardará
     *                  el número de píxeles que cumplen la condición (para 31..33).
     *                  Si es NULL, se ignora.
	 * @param haloSize  [in]  Tamaño del halo para el outline
     * @return          0 si se procesa correctamente; otro valor en caso de error.
     */
    int procImg(Pixel* pixels, int height, int width, int option, int filterDiv, unsigned int* outCount, int haloSize);

#ifdef __cplusplus
}
#endif

#endif // PROCIMG_H
