#include <stdio.h>

const char *ruta_def = "../img/img.bmp";

typedef struct {
  unsigned char r, g, b;
} Pixel;

int main() {
  printf("PECL 1 PAP \n \
Introduzca la ruta base de la imagen: \n \
(pulse intro para usar por defecto: ../img/img.bmp)");

  char ruta[100];
  scanf("%s", ruta);

  if (ruta[0] == 0) {
    printf("Ruta introducida: %s\n", ruta);
    ruta_def = ruta;
  }

  // Abrir archivo bmp y leer pixeles con librería para BMP

  printf("Opciones \n \
        (1) Conversión a Blanco y Negro \n \
        (2) Pixelar \n \
        (3) Identificar colores \n \
        (4) Filtro y delineado \n \
    ");

}
