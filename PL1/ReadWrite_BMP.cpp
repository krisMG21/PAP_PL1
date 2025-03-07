#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

#define MAGIC_VALUE         0x4D42 
#define NUM_PLANE           1
#define COMPRESSION         0
#define NUM_COLORS          0
#define IMPORTANT_COLORS    0
#define BITS_PER_PIXEL      24
#define BITS_PER_BYTE       8
#define BYTES_PER_PIXEL     BITS_PER_PIXEL/BITS_PER_BYTE

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

// Eliminamos RGB_t y utilizamos Pixel en su lugar

// Definimos el tipo de dato Pixel
typedef struct {
    unsigned char r, g, b;
} Pixel;

// Constante de color negro
const Pixel BLACK = { 0, 0, 0 };

static int getPadding(const BMPHeader_t* header) {
    return (4 - (header->width_px * BYTES_PER_PIXEL) % 4) % 4;
}

static int getRowSize(const BMPHeader_t* header) {
    return header->width_px * BYTES_PER_PIXEL + getPadding(header);
}

static int getPosition(const BMPHeader_t* header, int x, int y) {
    int j = (header->height_px - y - 1) * getRowSize(header);
    int i = x * BYTES_PER_PIXEL;
    return i + j;
}

int CheckHeader(const BMPHeader_t* header) {
    return header->type == MAGIC_VALUE
        && header->num_planes == NUM_PLANE
        && header->bits_per_pixel == BITS_PER_PIXEL
        && header->compression == COMPRESSION
        && header->num_colors == NUM_COLORS;
}

BMPImage_t* ReadBMP(const char* filename) {
    BMPImage_t* bmp = (BMPImage_t*)malloc(sizeof(BMPImage_t));

    FILE* image = fopen(filename, "rb");
    if (!image) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    if (fread(&bmp->header, sizeof(BMPHeader_t), 1, image) != 1) {
        std::cerr << "Cannot read image header" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!CheckHeader(&bmp->header)) {
        std::cerr << "Bad image" << std::endl;
        exit(EXIT_FAILURE);
    }

    bmp->data = (uint8_t*)malloc(bmp->header.image_size_bytes);
    if (!bmp->data) {
        std::cerr << "Cannot allocate memory" << std::endl;
        exit(EXIT_FAILURE);
    }

    fseek(image, bmp->header.offset, SEEK_SET);
    if (fread(bmp->data, bmp->header.image_size_bytes, 1, image) != 1) {
        std::cerr << "Cannot read image data" << std::endl;
        exit(EXIT_FAILURE);
    }

    fclose(image);
    return bmp;
}

void SaveBMP(const BMPImage_t* bmp, const char* filename) {
    FILE* fd = fopen(filename, "wb");
    if (!fd) {
        std::cerr << "Error writing file: " << filename << std::endl;
        return;
    }
    fwrite(&bmp->header, sizeof(BMPHeader_t), 1, fd);
    fseek(fd, bmp->header.offset, SEEK_SET);
    fwrite(bmp->data, bmp->header.image_size_bytes, 1, fd);
    fclose(fd);
}

void DestroyBMP(BMPImage_t* bmp) {
    if (bmp) {
        if (bmp->data) {
            free(bmp->data);
        }
        free(bmp);
    }
}

Pixel GetPixel(BMPImage_t* bmp, int x, int y) {
    if (x < 0 || x >= bmp->header.width_px || y < 0 || y >= bmp->header.height_px) {
        std::cerr << "Error: Pixel out of bounds!" << std::endl;
        return BLACK;
    }

    int pos = getPosition(&bmp->header, x, y);
    Pixel p;
    p.r = bmp->data[pos + 2];
    p.g = bmp->data[pos + 1];
    p.b = bmp->data[pos + 0];
    return p;
}

void SetPixel(BMPImage_t* bmp, int x, int y, Pixel pixel) {
    if (x < 0 || x >= bmp->header.width_px || y < 0 || y >= bmp->header.height_px) {
        std::cerr << "Error: Pixel out of bounds!" << std::endl;
        return;
    }

    int pos = getPosition(&bmp->header, x, y);
    bmp->data[pos + 2] = pixel.r;
    bmp->data[pos + 1] = pixel.g;
    bmp->data[pos + 0] = pixel.b;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <BMP file path>" << std::endl;
        return 1;
    }
    const char* filename = argv[1];
    BMPImage_t* bmp = ReadBMP(filename);

    Pixel black = { 0, 0, 0 };
    for (int i = 0; i < 1280; i++) {
        SetPixel(bmp, i, 10, black);
    }

    SaveBMP(bmp, "pic_out.bmp");
    DestroyBMP(bmp);

    return 0;
}
