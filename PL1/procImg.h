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
     * @brief Procesa la imagen usando CUDA según la opción dada.
     * @param pixels   Puntero al array lineal de píxeles.
     * @param height   Altura de la imagen (número de filas).
     * @param width    Anchura de la imagen (número de columnas).
     * @param option   Opción de procesamiento:
	 * @param tamFiltro Tamaño del filtro a aplicar.
     * @return 0 si se procesa correctamente; otro valor en caso de error.
     */
    int procImg(Pixel* pixels, int height, int width, int option, int tamFiltro);

#ifdef __cplusplus
}
#endif

#endif // PROCIMG_H
