#ifndef KERNEL_H
#define KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

// Definición de Pixel compartida entre C++ y CUDA
typedef struct {
    unsigned char r, g, b;
} Pixel;

/**
 * Función que procesa la imagen usando CUDA.
 * - pixels: puntero al array lineal de pixeles.
 * - nrows: número de filas de la imagen.
 * - ncolumns: número de columnas de la imagen.
 *
 * Retorna 0 en caso de éxito, otro valor si ocurre algún error.
 */
int procImg(Pixel* pixels, int nrows, int ncolumns);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_H
