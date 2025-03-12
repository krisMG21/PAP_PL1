#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include "procImg.h" // Donde se declara Pixel y la funcion procImg(...)

////////////////////////////////////////////////////////////////////////////////
// Kernel: Convierte un pixel a escala de grises
////////////////////////////////////////////////////////////////////////////////
__global__ void toGrayKernel(Pixel* d_pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float r = static_cast<float>(d_pixels[idx].r);
        float g = static_cast<float>(d_pixels[idx].g);
        float b = static_cast<float>(d_pixels[idx].b);

        float grayf = 0.299f * r + 0.587f * g + 0.114f * b;
        unsigned char gray = static_cast<unsigned char>(grayf);

        d_pixels[idx].r = gray;
        d_pixels[idx].g = gray;
        d_pixels[idx].b = gray;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Kernel: Invierte colores
// R = 255 - R, G = 255 - G, B = 255 - B
////////////////////////////////////////////////////////////////////////////////
__global__ void invertColorsKernel(Pixel* d_pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        d_pixels[idx].r = 255 - d_pixels[idx].r;
        d_pixels[idx].g = 255 - d_pixels[idx].g;
        d_pixels[idx].b = 255 - d_pixels[idx].b;
    }
}

// Kernel: Pixelado con filtro cuadrado de tamFiltro x tamFiltro
// Cada hilo copia no solo su pixel central, sino tambien parte del halo.
// Sin padding artificial: si la ventana se sale de [0..width-1], [0..height-1], no aporta nada (no se suma).
__global__ void pixelateKernel(const Pixel* d_in, Pixel* d_out,
    int width, int height, int tamFiltro)
{
    // Mitad del filtro
    int radius = tamFiltro / 2;

    // Coordenadas globales del hilo en la imagen
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    // Dimensiones del tile ampliado en shared memory (bloque + 2*halo)
    int tileW = blockDim.x + 2 * radius;
    int tileH = blockDim.y + 2 * radius;

    // Memoria compartida dinamica
    extern __shared__ Pixel sData[];  // tamaño tileW * tileH

    // Coordenadas locales dentro del tile
    // Dejas radius celdas a la izquierda/arriba para el halo
    int sX = threadIdx.x + radius;
    int sY = threadIdx.y + radius;

    // Cargar el pixel central
    if (gx < width && gy < height) {
        sData[sY * tileW + sX] = d_in[gy * width + gx];
    }
    else {
        sData[sY * tileW + sX] = { 0,0,0 };
    }

    // Cargar el halo a la izquierda (si threadIdx.x < radius)
    // Cada hilo que esté en la "columna" [0..radius-1] se encarga de traer esos pixeles (gx - i). 
    for (int i = 1; i <= radius; i++) {
        if (threadIdx.x < radius) {
            int haloXlocal = sX - i;   // sX - (1..radius)
            int haloXglobal = gx - i;  // gx - (1..radius)
            if (haloXlocal >= 0 && haloXglobal >= 0 && gy < height) {
                sData[sY * tileW + haloXlocal] = d_in[gy * width + haloXglobal];
            }
            else {
                sData[sY * tileW + haloXlocal] = { 0,0,0 };
            }
        }
    }

    // Cargar el halo a la derecha (si threadIdx.x >= blockDim.x - radius)
    for (int i = 1; i <= radius; i++) {
        int edgeX = blockDim.x - radius;
        if (threadIdx.x >= edgeX) {
            int haloXlocal = sX + i;   // sX + (1..radius)
            int haloXglobal = gx + i;  // gx + (1..radius)
            if (haloXlocal < tileW && haloXglobal < width && gy < height) {
                sData[sY * tileW + haloXlocal] = d_in[gy * width + haloXglobal];
            }
            else {
                sData[sY * tileW + haloXlocal] = { 0,0,0 };
            }
        }
    }

    // Cargar el halo superior (si threadIdx.y < radius)
    for (int j = 1; j <= radius; j++) {
        if (threadIdx.y < radius) {
            int haloYlocal = sY - j;
            int haloYglobal = gy - j;
            if (haloYlocal >= 0 && haloYglobal >= 0 && gx < width) {
                sData[haloYlocal * tileW + sX] = d_in[haloYglobal * width + gx];
            }
            else {
                sData[haloYlocal * tileW + sX] = { 0,0,0 };
            }
        }
    }

    // Cargar el halo inferior (si threadIdx.y >= blockDim.y - radius)
    for (int j = 1; j <= radius; j++) {
        int edgeY = blockDim.y - radius;
        if (threadIdx.y >= edgeY) {
            int haloYlocal = sY + j;
            int haloYglobal = gy + j;
            if (haloYlocal < tileH && haloYglobal < height && gx < width) {
                sData[haloYlocal * tileW + sX] = d_in[haloYglobal * width + gx];
            }
            else {
                sData[haloYlocal * tileW + sX] = { 0,0,0 };
            }
        }
    }

    // Sincronizar para asegurar que todo el tile se ha cargado
    __syncthreads();

    // ------------------------------------------------------------------------
    // Calcular el promedio en la ventana tamFiltro x tamFiltro
    // ------------------------------------------------------------------------
    if (gx < width && gy < height) {
        int sumR = 0, sumG = 0, sumB = 0;
        int count = 0;

        // La posicion local base
        for (int fy = -radius; fy <= radius; fy++) {
            for (int fx = -radius; fx <= radius; fx++) {
                int nx = sX + fx;  // coords en sData
                int ny = sY + fy;

                // Comprobar submatriz
                if (nx >= 0 && nx < tileW && ny >= 0 && ny < tileH) {
                    // Comprobar coords globales para no sumar pixeles fuera
                    int gx2 = gx + fx;
                    int gy2 = gy + fy;
                    if (gx2 >= 0 && gx2 < width && gy2 >= 0 && gy2 < height) {
                        Pixel pp = sData[ny * tileW + nx];
                        sumR += pp.r;
                        sumG += pp.g;
                        sumB += pp.b;
                        count++;
                    }
                }
            }
        }

        if (count > 0) {
            unsigned char outR = (unsigned char)(sumR / float(count));
            unsigned char outG = (unsigned char)(sumG / float(count));
            unsigned char outB = (unsigned char)(sumB / float(count));
            d_out[gy * width + gx] = { outR, outG, outB };
        }
        else {
            // Caso raro en bordes
            d_out[gy * width + gx] = { 0,0,0 };
        }
    }
}

// para la carga completa del halo.
int procImg(Pixel* pixels, int height, int width, int option, int tamFiltro)
{
    cudaError_t cudaStatus;
    Pixel* d_in = nullptr;
    Pixel* d_out = nullptr;
    int totalPixels = width * height;

    // 1) Info de la GPU
    int devID = 0;
    cudaDeviceProp prop;
    cudaStatus = cudaGetDeviceProperties(&prop, devID);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!\n");
        return 1;
    }

    // 2) Determinar blockDim.x = blockDim.y = bCandidate
    int maxThreads = prop.maxThreadsPerBlock;
    int bCandidate = static_cast<int>(floorf(std::sqrtf((float)maxThreads)));
    if (bCandidate > prop.maxThreadsDim[0]) bCandidate = prop.maxThreadsDim[0];
    if (bCandidate > prop.maxThreadsDim[1]) bCandidate = prop.maxThreadsDim[1];
    dim3 blockDim(bCandidate, bCandidate, 1);

    // 3) Calcular grid
    int gridX = (width + blockDim.x - 1) / blockDim.x;
    int gridY = (height + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridX, gridY, 1);

    printf("=== GPU Info ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("MaxThreadsPerBlock: %d, blockDim=(%d,%d), gridDim=(%d,%d)\n",
        maxThreads, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    printf("tamFiltro = %d\n", tamFiltro);
    printf("================\n");

    // 4) Seleccionar device
    cudaStatus = cudaSetDevice(devID);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\n");
        return 1;
    }

    // 5) Reservar memoria en GPU
    cudaStatus = cudaMalloc((void**)&d_in, totalPixels * sizeof(Pixel));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_in failed!\n");
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&d_out, totalPixels * sizeof(Pixel));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_out failed!\n");
        cudaFree(d_in);
        return 1;
    }

    // 6) Copiar la imagen host->device
    cudaStatus = cudaMemcpy(d_in, pixels, totalPixels * sizeof(Pixel), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H->D failed!\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    // 7) Segun la opcion
    switch (option) {
    case 1: // Blanco y Negro (in-place)
        toGrayKernel << <gridDim, blockDim >> > (d_in, width, height);
        break;

    case 6: // Invertir colores (in-place)
        invertColorsKernel << <gridDim, blockDim >> > (d_in, width, height);
        break;

    case 31: // Pixelar color
    {
        int radius = tamFiltro / 2;
        int tileW = blockDim.x + 2 * radius;
        int tileH = blockDim.y + 2 * radius;
        size_t needed = tileW * tileH * sizeof(Pixel);

        if (needed > prop.sharedMemPerBlock) {
            fprintf(stderr, "Error: tamFiltro=%d => %zu bytes en shared, pero solo hay %zu.\n",
                tamFiltro, needed, (size_t)prop.sharedMemPerBlock);
            cudaFree(d_in); cudaFree(d_out);
            return 1;
        }

        pixelateKernel << <gridDim, blockDim, needed >> > (d_in, d_out, width, height, tamFiltro);
    }
    break;

    case 32: // Pixelar BN (1) BN in-place, (2) pixelar out-of-place
    {
        toGrayKernel << <gridDim, blockDim >> > (d_in, width, height);
        cudaDeviceSynchronize();

        int radius = tamFiltro / 2;
        int tileW = blockDim.x + 2 * radius;
        int tileH = blockDim.y + 2 * radius;
        size_t needed = tileW * tileH * sizeof(Pixel);

        if (needed > prop.sharedMemPerBlock) {
            fprintf(stderr, "Error: tamFiltro=%d => %zu bytes en shared, solo hay %zu.\n",
                tamFiltro, needed, (size_t)prop.sharedMemPerBlock);
            cudaFree(d_in); cudaFree(d_out);
            return 1;
        }

        pixelateKernel << <gridDim, blockDim, needed >> > (d_in, d_out, width, height, tamFiltro);
    }
    break;

    default:
        fprintf(stderr, "Opcion %d no reconocida.\n", option);
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    // Verificar si hubo error lanzando el kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_in); cudaFree(d_out);
        return 1;
    }

    // Sincronizar
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error %d\n", cudaStatus);
        cudaFree(d_in); cudaFree(d_out);
        return 1;
    }

    // 8) Copiar el resultado de vuelta
    bool usedPixelate = (option == 31 || option == 32);
    Pixel* d_result = usedPixelate ? d_out : d_in;
    cudaStatus = cudaMemcpy(pixels, d_result, totalPixels * sizeof(Pixel), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D->H failed!\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    // 9) Liberar
    cudaFree(d_in);
    cudaFree(d_out);

    // 10) (Opcional) reset
    cudaDeviceReset();

    return 0; // exito
}
