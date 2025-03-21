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
    // Coordenadas globales del píxel
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    bool inside = (gx < width) && (gy < height);

    // Memoria compartida para la tesela y la reducción
    extern __shared__ int shared_mem[];
    Pixel* sData = (Pixel*)shared_mem;
    int* sR = (int*)(sData + blockDim.x * blockDim.y);
    int* sG = sR + blockDim.x * blockDim.y;
    int* sB = sG + blockDim.x * blockDim.y;
    int* sCount = sB + blockDim.x * blockDim.y;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Cargar píxel en memoria compartida
    if (inside) {
        sData[tid] = d_in[gy * width + gx];
    }
    else {
        sData[tid] = { 0, 0, 0 };
    }

    // Inicializar acumuladores
    sR[tid] = (inside) ? sData[tid].r : 0;
    sG[tid] = (inside) ? sData[tid].g : 0;
    sB[tid] = (inside) ? sData[tid].b : 0;
    sCount[tid] = (inside) ? 1 : 0;

    __syncthreads();

    // Reducción paralela para sumar R, G, B y contar píxeles válidos
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sR[tid] += sR[tid + stride];
            sG[tid] += sG[tid + stride];
            sB[tid] += sB[tid + stride];
            sCount[tid] += sCount[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 calcula el promedio y lo almacena en sData[0]
    if (tid == 0) {
        Pixel avg;
        if (sCount[0] > 0) {
            avg.r = (unsigned char)(sR[0] / sCount[0]);
            avg.g = (unsigned char)(sG[0] / sCount[0]);
            avg.b = (unsigned char)(sB[0] / sCount[0]);
        }
        else {
            avg = { 0, 0, 0 };
        }
        sData[0] = avg;
    }

    __syncthreads();

    // Todos los hilos escriben el promedio en su píxel
    if (inside) {
        d_out[gy * width + gx] = sData[0];
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
        /*int radius = tamFiltro / 2;
        int tileW = blockDim.x + 2 * radius;
        int tileH = blockDim.y + 2 * radius;*/
        /*size_t needed = tileW * tileH * sizeof(Pixel);
        size_t needed = blockSize.x * blockSize.y * (sizeof(Pixel) + 4 * sizeof(int));*/
        // Calcular la memoria compartida necesaria para pixelateKernel
        int nThreads = blockDim.x * blockDim.y;

        // Tamaño para los pixeles
        size_t sizeData = nThreads * sizeof(Pixel);

        // Tamaño para R, G, B, Count (4 int arrays)
        size_t sizeR = nThreads * sizeof(int);
        size_t sizeG = nThreads * sizeof(int);
        size_t sizeB = nThreads * sizeof(int);
        size_t sizeCount = nThreads * sizeof(int);

        // Total
        size_t totalNeeded = sizeData + sizeR + sizeG + sizeB + sizeCount;


        if (totalNeeded > prop.sharedMemPerBlock) {
            fprintf(stderr, "Error: tamFiltro=%d => %zu bytes en shared, pero solo hay %zu.\n",
                tamFiltro, totalNeeded, (size_t)prop.sharedMemPerBlock);
            cudaFree(d_in); cudaFree(d_out);
            return 1;
        }

        pixelateKernel << <gridDim, blockDim, totalNeeded >> > (d_in, d_out, width, height, tamFiltro);
    }
    break;

    case 32: // Pixelar BN (1) BN in-place, (2) pixelar out-of-place
    {
        toGrayKernel << <gridDim, blockDim >> > (d_in, width, height);
        cudaDeviceSynchronize();

        int nThreads = blockDim.x * blockDim.y;

        // Tamaño para los pixeles
        size_t sizeData = nThreads * sizeof(Pixel);

        // Tamaño para R, G, B, Count (4 int arrays)
        size_t sizeR = nThreads * sizeof(int);
        size_t sizeG = nThreads * sizeof(int);
        size_t sizeB = nThreads * sizeof(int);
        size_t sizeCount = nThreads * sizeof(int);

        // Total
        size_t totalNeeded = sizeData + sizeR + sizeG + sizeB + sizeCount;

        if (totalNeeded > prop.sharedMemPerBlock) {
            fprintf(stderr, "Error: tamFiltro=%d => %zu bytes en shared, solo hay %zu.\n",
                tamFiltro, totalNeeded, (size_t)prop.sharedMemPerBlock);
            cudaFree(d_in); cudaFree(d_out);
            return 1;
        }

        pixelateKernel << <gridDim, blockDim, totalNeeded >> > (d_in, d_out, width, height, tamFiltro);
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
