#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include "procImg.h"

#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

#pragma pack(push, 1)
struct BMPHeader_t {
    uint16_t  type;
    uint32_t  size;
    uint16_t  reserved1;
    uint16_t  reserved2;
    uint32_t  offset;
    uint32_t  dib_header_size;
    int32_t   width_px;
    int32_t   height_px;
    uint16_t  num_planes;
    uint16_t  bits_per_pixel;
    uint32_t  compression;
    uint32_t  image_size_bytes;
    int32_t   x_resolution_ppm;
    int32_t   y_resolution_ppm;
    uint32_t  num_colors;
    uint32_t  important_colors;
};
#pragma pack(pop)

struct BMPImage_t {
    BMPHeader_t header;
    uint8_t* data;
};

#define MAGIC_VALUE         0x4D42
#define BITS_PER_PIXEL      24
#define BYTES_PER_PIXEL     (BITS_PER_PIXEL / 8)

static int getPadding(const BMPHeader_t* header) {
    return (4 - (header->width_px * BYTES_PER_PIXEL) % 4) % 4;
}
static int getRowSize(const BMPHeader_t* header) {
    return header->width_px * BYTES_PER_PIXEL + getPadding(header);
}
static int getPosition(const BMPHeader_t* header, int x, int y) {
    int stride = getRowSize(header);
    int j = (header->height_px - y - 1) * stride;
    int i = x * BYTES_PER_PIXEL;
    return j + i;
}
static int CheckHeader(const BMPHeader_t* header) {
    return header->type == MAGIC_VALUE &&
        header->bits_per_pixel == BITS_PER_PIXEL;
}

// Leer BMP
BMPImage_t* ReadBMP(const char* filename) {
    BMPImage_t* bmp = (BMPImage_t*)malloc(sizeof(BMPImage_t));
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error abriendo archivo BMP: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fread(&bmp->header, sizeof(BMPHeader_t), 1, fp);
    if (!CheckHeader(&bmp->header)) {
        printf("Cabecera BMP no valida.\n");
        exit(EXIT_FAILURE);
    }
    bmp->data = (uint8_t*)malloc(bmp->header.image_size_bytes);
    fseek(fp, bmp->header.offset, SEEK_SET);
    fread(bmp->data, bmp->header.image_size_bytes, 1, fp);
    fclose(fp);
    return bmp;
}

// Guardar BMP
void SaveBMP(BMPImage_t* bmp, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error guardando archivo BMP: %s\n", filename);
        return;
    }
    fwrite(&bmp->header, sizeof(BMPHeader_t), 1, fp);
    fseek(fp, bmp->header.offset, SEEK_SET);
    fwrite(bmp->data, bmp->header.image_size_bytes, 1, fp);
    fclose(fp);
}

// Liberar
void DestroyBMP(BMPImage_t* bmp) {
    if (bmp) {
        free(bmp->data);
        free(bmp);
    }
}

// Convierte bmp->data (con padding) a un array lineal de Pixel (sin padding)
std::vector<Pixel> bmpToPixelArray(const BMPImage_t* bmp) {
    int width = bmp->header.width_px;
    int height = bmp->header.height_px;
    std::vector<Pixel> out;
    out.reserve(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pos = getPosition(&bmp->header, x, y);
            Pixel p;
            p.b = bmp->data[pos + 0];
            p.g = bmp->data[pos + 1];
            p.r = bmp->data[pos + 2];
            out.push_back(p);
        }
    }
    return out;
}

// Copia de vuelta
void pixelArrayToBMP(BMPImage_t* bmp, const std::vector<Pixel>& in) {
    int width = bmp->header.width_px;
    int height = bmp->header.height_px;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pos = getPosition(&bmp->header, x, y);
            int idx = y * width + x;
            bmp->data[pos + 0] = in[idx].b;
            bmp->data[pos + 1] = in[idx].g;
            bmp->data[pos + 2] = in[idx].r;
        }
    }
}

int main() {
    char ruta[256];
    memset(ruta, 0, sizeof(ruta));
    const char* defaultFile = "test.bmp";
    printf("Introduzca la ruta de la imagen BMP (Intro para usar '%s'): ", defaultFile);
    if (fgets(ruta, sizeof(ruta), stdin) == nullptr) {
        strcpy(ruta, defaultFile);
    }
    else {
        size_t len = strlen(ruta);
        if (len > 0 && ruta[len - 1] == '\n')
            ruta[len - 1] = '\0';
        if (strlen(ruta) == 0)
            strcpy(ruta, defaultFile);
    }

    printf("Leyendo archivo: '%s'\n", ruta);
    BMPImage_t* bmp = ReadBMP(ruta);
    int width = bmp->header.width_px;
    int height = bmp->header.height_px;
    printf("Imagen de %d x %d pixeles\n", width, height);

    std::vector<Pixel> pixels = bmpToPixelArray(bmp);

    printf("Opciones\n"
        "  (1) Conversion a Blanco y Negro\n"
        "  (2) Pixelar\n"
        "  (3) Identificar colores (sin halo)\n"
        "  (4) Filtro y delineado de zonas de color\n"
        "  (5) Calculo de pseudo-hash\n"
        "  (6) Invertir colores\n"
        "  (X) Salir\n");
    printf("Elija opcion: ");

    int option;
    scanf("%d", &option);

    int filterDiv = 1;
    const char* outName = nullptr;

    switch (option) {
    case 1:
        procImg(pixels.data(), height, width, 1, filterDiv, nullptr, 0);
        outName = "out_gray.bmp";
        break;

    case 2: {
        printf("Has elegido pixelar: \n (1) Color \n (2) Blanco y Negro\n");
        int subSel;
        scanf("%d", &subSel);
        printf("Introduce factor de division del blockDim: ");
        scanf("%d", &filterDiv);
        if (subSel == 1) {
            procImg(pixels.data(), height, width, 21, filterDiv, nullptr, 0);
            outName = "out_pixel_color.bmp";
        }
        else if (subSel == 2) {
            procImg(pixels.data(), height, width, 22, filterDiv, nullptr, 0);
            outName = "out_pixel_bn.bmp";
        }
        else {
            printf("Seleccion no valida. Saliendo...\n");
            DestroyBMP(bmp);
            return 0;
        }
    }
          break;

    case 3: {
        // Identificaci�n sin halo
        unsigned int count = 0;
        printf("Elija el color a identificar (1: Rojo, 2: Verde, 3: Azul): ");
        int colorSel;
        scanf("%d", &colorSel);
        int optIdent;
        if (colorSel == 1)
            optIdent = 31;
        else if (colorSel == 2)
            optIdent = 32;
        else if (colorSel == 3)
            optIdent = 33;
        else {
            printf("Color no valido. Saliendo...\n");
            DestroyBMP(bmp);
            return 0;
        }
        procImg(pixels.data(), height, width, optIdent, 0, &count, 0);
        if (colorSel == 1)
            outName = "out_red.bmp";
        else if (colorSel == 2)
            outName = "out_green.bmp";
        else
            outName = "out_blue.bmp";
        printf("Num pixeles identificados = %u\n", count);
    }
          break;

    case 4: {
        // Filtro y delineado de zonas de color: se usan las opciones 41,42,43 según el color
        printf("Has elegido filtro y delineado.\n");
        printf("Elija el color a delinear (1: Rojo, 2: Verde, 3: Azul): ");
        int colorChoice;
        scanf("%d", &colorChoice);
        int optDelineado;
        if (colorChoice == 1)
            optDelineado = 41;
        else if (colorChoice == 2)
            optDelineado = 42;
        else if (colorChoice == 3)
            optDelineado = 43;
        else {
            printf("Color no valido. Saliendo...\n");
            DestroyBMP(bmp);
            return 0;
        }
        printf("Introduce factor de division del blockDim: ");
        scanf("%d", &filterDiv);
        printf("Introduzca el tama�o del halo: ");
        int haloSize;
        scanf("%d", &haloSize);
        procImg(pixels.data(), height, width, optDelineado, filterDiv, nullptr, haloSize);
        outName = "out_filtro_delineado.bmp";
    }
          break;

    case 5: {
        // Pseudo-hash: calcular y mostrar el hash en consola
        // En esta opci�n, procImg realiza la reducci�n y muestra el hash textual.
        procImg(pixels.data(), height, width, 5, filterDiv, nullptr, 0);
        // No se guarda imagen; se asume que el hash se muestra por consola.
        printf("Hash calculado. Consulte la salida en consola.\n");
        // Salir inmediatamente
        DestroyBMP(bmp);
        return 0;
    }
          break;

    case 6: {
        procImg(pixels.data(), height, width, 6, filterDiv, nullptr, 0);
        outName = "out_inverted.bmp";
    }
        break;

    default:
        printf("Opcion no valida. Saliendo...\n");
        DestroyBMP(bmp);
        return 0;
    }

    if (!outName) {
        outName = "out_result.bmp";
    }

    pixelArrayToBMP(bmp, pixels);
    SaveBMP(bmp, outName);
    printf("Guardado resultado en: %s\n", outName);

    DestroyBMP(bmp);
    return 0;
}
