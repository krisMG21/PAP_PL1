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
    return header->type == MAGIC_VALUE
        && header->bits_per_pixel == BITS_PER_PIXEL;
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
    // Pedimos al usuario la ruta de la imagen (con valor por defecto)
    char ruta[256];
    memset(ruta, 0, sizeof(ruta));
    const char* defaultFile = "test.bmp";  // Nombre por defecto

    printf("Introduzca la ruta de la imagen BMP (Intro para usar '%s'): ", defaultFile);
    if (fgets(ruta, sizeof(ruta), stdin) == nullptr) {
        // Si ocurre algun error al leer, usamos la ruta por defecto
        strcpy(ruta, defaultFile);
    }
    else {
        // Eliminar el salto de linea
        size_t len = strlen(ruta);
        if (len > 0 && ruta[len - 1] == '\n') {
            ruta[len - 1] = '\0';
        }
        if (strlen(ruta) == 0) {
            // Si el usuario no tecleo nada (intro directo), usamos la ruta por defecto
            strcpy(ruta, defaultFile);
        }
    }

    printf("Leyendo archivo: '%s'\n", ruta);
    BMPImage_t* bmp = ReadBMP(ruta);
    int width = bmp->header.width_px;
    int height = bmp->header.height_px;
    printf("Imagen de %d x %d pixeles\n", width, height);

    // Convertir a array lineal de Pixel
    std::vector<Pixel> pixels = bmpToPixelArray(bmp);

    // Mostrar menu
    printf("Opciones \n \
        (1) Conversion a Blanco y Negro \n \
        (2) Pixelar \n \
        (3) Identificar colores (rojo,verde,azul) \n \
        (4) Filtro y delineado \n \
        (5) Calculo de pseudo-hash \n \
        (6) Invertir colores \n \
        (X) Salir\n");
    printf("Elija opcion: ");

    int option;
    scanf("%d", &option);

    // Valor por defecto "filterDiv" = 1
    // (si pixelar -> se pide tamFiltro, etc.)
    int filterDiv = 1;
    const char* outName = nullptr;

    switch (option) {
    case 1:
        // BN
        procImg(pixels.data(), height, width, 1, filterDiv, nullptr);
        outName = "out_gray.bmp";
        break;

    case 2:
    {
        // Submenu pixelar
        // (Lo que ya tenias: se llama (31) Pixelar color, o (32) Pixelar BN)
        printf("Has elegido pixelar: \n (1) Color \n (2) Blanco y Negro\n");
        int subSel;
        scanf("%d", &subSel);
        if (subSel == 1) {
            // Pixelar color => finalOption=31
            // Pides filterDiv o tamFiltro:
            printf("Introduce factor de division del blockDim: ");
            scanf("%d", &filterDiv);
            procImg(pixels.data(), height, width, 31, filterDiv, nullptr);
            outName = "out_pixel_color.bmp";
        }
        else if (subSel == 2) {
            // Pixelar BN => finalOption=32
            printf("Introduce factor de division del blockDim: ");
            scanf("%d", &filterDiv);
            procImg(pixels.data(), height, width, 32, filterDiv, nullptr);
            outName = "out_pixel_bn.bmp";
        }
        else {
            printf("Seleccion no valida. Saliendo...\n");
            DestroyBMP(bmp);
            return 0;
        }
    }
    break;

    case 3:
    {
        // Identificar colores => Llamamos 3 veces
        // 1) Fijar umbrales (o pedirlos al usuario)
        int hostRed[6] = { 100,255,   0,150,   0,150 };
        int hostGreen[6] = { 30,150,    50,255,  0,75 };
        int hostBlue[6] = { 0,200,     0,249,   100,255 };

        setRedThresholds(hostRed);
        setGreenThresholds(hostGreen);
        setBlueThresholds(hostBlue);

        // a) Identificar ROJO => opcion=41
        unsigned int countR = 0;
        procImg(pixels.data(), height, width, 41, 0, &countR);
        // Guardar out_red.bmp
        pixelArrayToBMP(bmp, pixels);
        SaveBMP(bmp, "out_red.bmp");
        printf("Num pixeles rojos = %u\n", countR);

        // Recargar la imagen original para no partir de la imagen ya modificada
        DestroyBMP(bmp);
        bmp = ReadBMP(ruta);
        pixels = bmpToPixelArray(bmp);

        // b) Identificar VERDE => opcion=42
        unsigned int countG = 0;
        procImg(pixels.data(), height, width, 42, 0, &countG);
        pixelArrayToBMP(bmp, pixels);
        SaveBMP(bmp, "out_green.bmp");
        printf("Num pixeles verdes = %u\n", countG);

        DestroyBMP(bmp);
        bmp = ReadBMP(ruta);
        pixels = bmpToPixelArray(bmp);

        // c) Identificar AZUL => opcion=43
        unsigned int countB = 0;
        procImg(pixels.data(), height, width, 43, 0, &countB);
        pixelArrayToBMP(bmp, pixels);
        SaveBMP(bmp, "out_blue.bmp");
        printf("Num pixeles azules = %u\n", countB);

        // Salimos
        DestroyBMP(bmp);
        return 0;
    }
    break;

    case 4:
        printf("Opcion 4: Filtro y delineado (no implementado).\n");
        outName = "out_filtro_delineado.bmp";
        break;

    case 5:
        printf("Opcion 5: Calculo de pseudo-hash (no implementado).\n");
        outName = "out_hash.bmp";
        break;

    case 6:
        // Invertir colores
        procImg(pixels.data(), height, width, 6, filterDiv, nullptr);
        outName = "out_inverted.bmp";
        break;

    default:
        printf("Opcion no valida. Saliendo...\n");
        DestroyBMP(bmp);
        return 0;
    }

    // Si la opcion no era 3 (que sale antes)
    if (!outName) {
        outName = "out_result.bmp";
    }

    // Copiamos el resultado de vuelta al BMP
    pixelArrayToBMP(bmp, pixels);

    // Guardar el resultado
    SaveBMP(bmp, outName);
    printf("Guardado resultado en: %s\n", outName);

    // Liberamos
    DestroyBMP(bmp);
    return 0;
}
