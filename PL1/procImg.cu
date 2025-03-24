#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include "procImg.h" // Donde se declara Pixel y la funcion procImg(...)

// -----------------------------------------------------------------------------
// Memoria de constantes para los umbrales de color
// Cada color (rojo, verde, azul) tendrá 6 enteros [Rmin,Rmax, Gmin,Gmax, Bmin,Bmax]
// -----------------------------------------------------------------------------
__constant__ int c_threshRed[6];
__constant__ int c_threshGreen[6];
__constant__ int c_threshBlue[6];

// -----------------------------------------------------------------------------
// Funciones para copiar umbrales desde host a device (const memory)
// -----------------------------------------------------------------------------
cudaError_t setRedThresholds(const int hostThresh[6]) {
    return cudaMemcpyToSymbol(c_threshRed, hostThresh, 6 * sizeof(int));
}

cudaError_t setGreenThresholds(const int hostThresh[6]) {
    return cudaMemcpyToSymbol(c_threshGreen, hostThresh, 6 * sizeof(int));
}

cudaError_t setBlueThresholds(const int hostThresh[6]) {
    return cudaMemcpyToSymbol(c_threshBlue, hostThresh, 6 * sizeof(int));
}

// -----------------------------------------------------------------------------
// Kernel: Convierte un pixel a escala de grises (in-place)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Kernel: Invierte colores (in-place)
// R = 255 - R, G = 255 - G, B = 255 - B
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Kernel: Pixelado con filtro basado en blockDim.x * blockDim.y
// (Sin padding artificial).
// Cada bloque calcula el promedio de todos los píxeles que lo conforman.
// -----------------------------------------------------------------------------
__global__ void pixelateKernel(const Pixel* d_in, Pixel* d_out, int width, int height)
{
    // Coordenadas globales del píxel
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    bool inside = (gx < width) && (gy < height);

    // Memoria compartida para la tesela (bloque) y la reducción
    extern __shared__ int shared_mem[];
    Pixel* sData = (Pixel*)shared_mem;
    int* sR = (int*)(sData + blockDim.x * blockDim.y);
    int* sG = sR + (blockDim.x * blockDim.y);
    int* sB = sG + (blockDim.x * blockDim.y);
    int* sCount = sB + (blockDim.x * blockDim.y);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Cargar píxel en memoria compartida
    if (inside) {
        sData[tid] = d_in[gy * width + gx];
    }
    else {
        // Si está fuera, lo ponemos a 0
        sData[tid] = { 0, 0, 0 };
    }

    // Inicializar acumuladores
    sR[tid] = (inside) ? sData[tid].r : 0;
    sG[tid] = (inside) ? sData[tid].g : 0;
    sB[tid] = (inside) ? sData[tid].b : 0;
    sCount[tid] = (inside) ? 1 : 0;

    __syncthreads();

    // Reducción paralela para sumar R, G, B y contar píxeles válidos
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sR[tid] += sR[tid + stride];
            sG[tid] += sG[tid + stride];
            sB[tid] += sB[tid + stride];
            sCount[tid] += sCount[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 del bloque calcula el promedio y lo almacena en sData[0]
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

// -----------------------------------------------------------------------------
// Kernel: Identificación de colores (out-of-place).
// thrColor en memoria constante (c_threshRed, c_threshGreen, c_threshBlue).
// Se hace atomicAdd a d_countColor cuando un píxel cumple la condición.
// -----------------------------------------------------------------------------
__device__ bool checkColor(const Pixel& px, const int thr[6])
{
    // thr: [Rmin,Rmax, Gmin,Gmax, Bmin,Bmax]
    if (px.r < thr[0] || px.r > thr[1]) return false;
    if (px.g < thr[2] || px.g > thr[3]) return false;
    if (px.b < thr[4] || px.b > thr[5]) return false;
    return true;
}

__global__ void identifyKernel(const Pixel* d_in, Pixel* d_out,
    int width, int height,
    const int* thrColor,
    unsigned int* d_countColor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        Pixel p = d_in[idx];

        bool isColor = checkColor(p, thrColor);
        if (isColor) {
            // Mantener color
            d_out[idx] = p;
            // Atomically incrementar contador
            atomicAdd(d_countColor, 1u);
        }
        else {
            // Pintar en blanco
            d_out[idx] = { 255, 255, 255 };
        }
    }
}
void setColorThresholds() {
    // Identificar colores => Llamamos 3 veces
    // 1) Fijar umbrales (o pedirlos al usuario)
    int hostRed[6] = { 100,255,   0,150,   0,150 };
    int hostGreen[6] = { 30,150,    50,255,  0,75 };
    int hostBlue[6] = { 0,200,     0,249,   100,255 };
    setRedThresholds(hostRed);
    setGreenThresholds(hostGreen);
    setBlueThresholds(hostBlue);
}

// -----------------------------------------------------------------------------
// Función principal para procesar la imagen
// -----------------------------------------------------------------------------
int procImg(Pixel* pixels, int height, int width,
    int option, int filterDiv, unsigned int* outCount)
{
    cudaError_t cudaStatus;
    Pixel* d_in = nullptr;
    Pixel* d_out = nullptr;
    int totalPixels = width * height;

    // Si el usuario pasa un outCount, lo iniciamos a 0
    if (outCount) *outCount = 0;

    // 1) Obtener info de la GPU (para blockDim / gridDim)
    int devID = 0;
    cudaDeviceProp prop;
    cudaStatus = cudaGetDeviceProperties(&prop, devID);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!\n");
        return 1;
    }

    // 2) Seleccionar la GPU
    cudaStatus = cudaSetDevice(devID);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\n");
        return 1;
    }

    // 3) Determinar blockDim.x = blockDim.y = bCandidate
    int maxThreads = prop.maxThreadsPerBlock;
    int bCandidate = static_cast<int>(floorf(std::sqrtf((float)maxThreads)));
    if (bCandidate > prop.maxThreadsDim[0]) bCandidate = prop.maxThreadsDim[0];
    if (bCandidate > prop.maxThreadsDim[1]) bCandidate = prop.maxThreadsDim[1];

    // Valor inicial de blockDim
    dim3 blockDim(bCandidate, bCandidate, 1);
    // Calculo inicial de grid
    int gridX = (width + blockDim.x - 1) / blockDim.x;
    int gridY = (height + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridX, gridY, 1);

    // Asegurarnos de que filterDiv sea valido para no obtener blockDim=0
    if (filterDiv < 1) filterDiv = 1;
    if (filterDiv > bCandidate) filterDiv = bCandidate;

    printf("=== GPU Info ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("MaxThreadsPerBlock: %d\n", maxThreads);
    printf("Initial blockDim=(%d,%d), gridDim=(%d,%d)\n",
        blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    printf("filterDiv = %d\n", filterDiv);
    printf("================\n");

    // 4) Reservar memoria en GPU
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

    // 5) Copiar la imagen host->device
    cudaStatus = cudaMemcpy(d_in, pixels, totalPixels * sizeof(Pixel), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy Host->Device failed!\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    // 6) Si necesitamos contadores (para identificar colores)
    unsigned int* d_count = nullptr;
    if (option == 41 || option == 42 || option == 43) {
        cudaStatus = cudaMalloc((void**)&d_count, sizeof(unsigned int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_count failed!\n");
            cudaFree(d_in);
            cudaFree(d_out);
            return 1;
        }
        unsigned int zero = 0;
        cudaStatus = cudaMemcpy(d_count, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy d_count init failed!\n");
            cudaFree(d_in);
            cudaFree(d_out);
            cudaFree(d_count);
            return 1;
        }
    }

    // Variable para saber si el resultado final esta en d_out
    bool devResultIn_d_out = false;

    // 7) Lanzar el kernel segun la opcion
    switch (option) {
    case 1: // Blanco y Negro => in-place
        toGrayKernel<<<gridDim, blockDim>>>(d_in, width, height);
        devResultIn_d_out = false;
        break;

    case 6: // Invertir colores => in-place
        invertColorsKernel<<<gridDim, blockDim>>>(d_in, width, height);
        devResultIn_d_out = false;
        break;

    case 21: // Pixelar en color => out-of-place
    {
        // Recalcular blockDim segun filterDiv para pixelado
        int newBx = bCandidate / filterDiv;
        int newBy = bCandidate / filterDiv;
        if (newBx < 1) newBx = 1;
        if (newBy < 1) newBy = 1;
        blockDim.x = newBx;
        blockDim.y = newBy;
        gridX = (width + newBx - 1) / newBx;
        gridY = (height + newBy - 1) / newBy;
        gridDim.x = gridX;
        gridDim.y = gridY;
        // Calcular memoria compartida necesaria
        int nThreads = newBx * newBy;
        size_t sizeData  = nThreads * sizeof(Pixel);
        size_t sizeR     = nThreads * sizeof(int);
        size_t sizeG     = nThreads * sizeof(int);
        size_t sizeB     = nThreads * sizeof(int);
        size_t sizeCount = nThreads * sizeof(int);
        size_t totalNeeded = sizeData + sizeR + sizeG + sizeB + sizeCount;
        pixelateKernel<<<gridDim, blockDim, totalNeeded>>>(d_in, d_out, width, height);
        devResultIn_d_out = true;
    }
    break;

    case 22: // Pixelar BN => primero BN in-place, luego pixelado out-of-place
    {
        toGrayKernel<<<gridDim, blockDim>>>(d_in, width, height);
        cudaDeviceSynchronize();
        int newBx = bCandidate / filterDiv;
        int newBy = bCandidate / filterDiv;
        if (newBx < 1) newBx = 1;
        if (newBy < 1) newBy = 1;
        blockDim.x = newBx;
        blockDim.y = newBy;
        gridX = (width + newBx - 1) / newBx;
        gridY = (height + newBy - 1) / newBy;
        gridDim.x = gridX;
        gridDim.y = gridY;
        int nThreads = newBx * newBy;
        size_t sizeData  = nThreads * sizeof(Pixel);
        size_t sizeR     = nThreads * sizeof(int);
        size_t sizeG     = nThreads * sizeof(int);
        size_t sizeB     = nThreads * sizeof(int);
        size_t sizeCount = nThreads * sizeof(int);
        size_t totalNeeded = sizeData + sizeR + sizeG + sizeB + sizeCount;
        pixelateKernel<<<gridDim, blockDim, totalNeeded>>>(d_in, d_out, width, height);
        devResultIn_d_out = true;
    }
    break;

    case 31: // Identificar Rojo => out-of-place
    {
		setColorThresholds();
        // Obtener dirección del símbolo de constantes
        const int* pRed = nullptr;
        cudaStatus = cudaGetSymbolAddress((void**)&pRed, c_threshRed);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetSymbolAddress (c_threshRed) failed: %s\n", cudaGetErrorString(cudaStatus));
            cudaFree(d_in); cudaFree(d_out);
            return 1;
        }
        identifyKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height, pRed, d_count);
        devResultIn_d_out = true;
    }
    break;

    case 32: // Identificar Verde => out-of-place
    {
		setColorThresholds();
        const int* pGreen = nullptr;
        cudaStatus = cudaGetSymbolAddress((void**)&pGreen, c_threshGreen);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetSymbolAddress (c_threshGreen) failed: %s\n", cudaGetErrorString(cudaStatus));
            cudaFree(d_in); cudaFree(d_out);
            return 1;
        }
        identifyKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height, pGreen, d_count);
        devResultIn_d_out = true;
    }
    break;

    case 33: // Identificar Azul => out-of-place
    {
        setColorThresholds();
        const int* pBlue = nullptr;
        cudaStatus = cudaGetSymbolAddress((void**)&pBlue, c_threshBlue);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetSymbolAddress (c_threshBlue) failed: %s\n", cudaGetErrorString(cudaStatus));
            cudaFree(d_in); cudaFree(d_out);
            return 1;
        }
        identifyKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height, pBlue, d_count);
        devResultIn_d_out = true;
    }
    break;

    default:
        fprintf(stderr, "Opcion %d no reconocida.\n", option);
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    // 9) Verificar errores de lanzamiento
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

    // 10) Copiar el resultado de vuelta a "pixels"
    //     - BN e Invertir => in-place => d_in
    //     - Pixelar / Identificar color => out-of-place => d_out
    Pixel* d_result = devResultIn_d_out ? d_out : d_in;
    cudaStatus = cudaMemcpy(pixels, d_result, totalPixels * sizeof(Pixel), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D->H failed!\n");
        cudaFree(d_in); cudaFree(d_out);
        return 1;
    }

    // 11) Si es Identificar colores (41,42,43), leer el contador
    if (option == 41 || option == 42 || option == 43) {
        unsigned int hCount = 0;
        cudaMemcpy(&hCount, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (outCount) *outCount = hCount;
        cudaFree(d_count);
    }

    // 12) Liberar
    cudaFree(d_in);
    cudaFree(d_out);

    // 13) (Opcional) reset.
    //     Si tu main va a llamar procImg varias veces, es aconsejable
    //     comentar esta línea para no perder el contexto:
    // cudaDeviceReset();

    return 0; // éxito
}
