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
     *                     - 31: Pixelar en color
     *                     - 32: Pixelar en B/N
     *                     - 41: Identificar color rojo
     *                     - 42: Identificar color verde
     *                     - 43: Identificar color azul
     * @param filterDiv [in]  Factor para redimensionar el blockDim (ej.: si pixelar).
     * @param outCount  [out] (Opcional) Puntero a variable donde se guardará
     *                  el número de píxeles que cumplen la condición (para 41..43).
     *                  Si es NULL, se ignora.
     * @return          0 si se procesa correctamente; otro valor en caso de error.
     */
    int procImg(Pixel* pixels, int height, int width, int option, int filterDiv, unsigned int* outCount);

    /**
     * @brief Copia a la memoria de constantes los umbrales de ROJO.
     * @param hostThresh  Array de 6 enteros: {Rmin,Rmax, Gmin,Gmax, Bmin,Bmax}.
     * @return cudaError_t
     */
    cudaError_t setRedThresholds(const int hostThresh[6]);

    /**
     * @brief Copia a la memoria de constantes los umbrales de VERDE.
     * @param hostThresh  Array de 6 enteros: {Rmin,Rmax, Gmin,Gmax, Bmin,Bmax}.
     * @return cudaError_t
     */
    cudaError_t setGreenThresholds(const int hostThresh[6]);

    /**
     * @brief Copia a la memoria de constantes los umbrales de AZUL.
     * @param hostThresh  Array de 6 enteros: {Rmin,Rmax, Gmin,Gmax, Bmin,Bmax}.
     * @return cudaError_t
     */
    cudaError_t setBlueThresholds(const int hostThresh[6]);

#ifdef __cplusplus
}
#endif

#endif // PROCIMG_H
