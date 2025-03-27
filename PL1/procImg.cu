#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include "procImg.h" // Se declara Pixel y la función procImg(...)


// -----------------------------------------------------------------------------
// Memoria de constantes para los umbrales de color
// Cada color (rojo, verde, azul) tendrá 6 enteros: [Rmin, Rmax, Gmin, Gmax, Bmin, Bmax]
// -----------------------------------------------------------------------------
__constant__ int c_threshRed[6];
__constant__ int c_threshGreen[6];
__constant__ int c_threshBlue[6];
int* reduction[];

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
// Función auxiliar para fijar thresholds de color de forma fija (se invoca en identificación)
// -----------------------------------------------------------------------------
void setColorThresholds() {
    int hostRed[6] = { 100, 255,   0, 150,   0, 150 };
    int hostGreen[6] = { 30,  150,   50, 255,   0, 75 };
    int hostBlue[6] = { 0,   200,    0, 249,  100, 255 };
    setRedThresholds(hostRed);
    setGreenThresholds(hostGreen);
    setBlueThresholds(hostBlue);
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
// Kernel: Pixelado (promedio de la tesela)
// -----------------------------------------------------------------------------
__global__ void pixelateKernel(const Pixel* d_in, Pixel* d_out, int width, int height)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    bool inside = (gx < width) && (gy < height);
    extern __shared__ int shared_mem[];
    Pixel* sData = (Pixel*)shared_mem;
    int* sR = (int*)(sData + blockDim.x * blockDim.y);
    int* sG = sR + (blockDim.x * blockDim.y);
    int* sB = sG + (blockDim.x * blockDim.y);
    int* sCount = sB + (blockDim.x * blockDim.y);
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (inside)
        sData[tid] = d_in[gy * width + gx];
    else
        sData[tid] = { 0, 0, 0 };
    sR[tid] = (inside) ? sData[tid].r : 0;
    sG[tid] = (inside) ? sData[tid].g : 0;
    sB[tid] = (inside) ? sData[tid].b : 0;
    sCount[tid] = (inside) ? 1 : 0;
    __syncthreads();
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sR[tid] += sR[tid + stride];
            sG[tid] += sG[tid + stride];
            sB[tid] += sB[tid + stride];
            sCount[tid] += sCount[tid + stride];
        }
        __syncthreads();
    }
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
    if (inside)
        d_out[gy * width + gx] = sData[0];
}

// -----------------------------------------------------------------------------
// Kernel: Identificación de colores sin halo (out-of-place).
// Usa el umbral correspondiente (31: rojo, 32: verde, 33: azul)
// -----------------------------------------------------------------------------
__device__ bool checkColor(const Pixel& px, const int thr[6])
{
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
            d_out[idx] = p;
            atomicAdd(d_countColor, 1u);
        }
        else {
            d_out[idx] = { 255, 255, 255 };
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: Delineado básico.
// Para cada píxel no blanco en la imagen identificada, revisa una ventana
// de tamaño (2*haloSize+1) x (2*haloSize+1). Si se detecta al menos un vecino blanco,
// pinta el píxel actual de negro (halo).
// -----------------------------------------------------------------------------
// Kernel: Delineado básico (outline hacia afuera).
// Para cada píxel blanco en la imagen identificada, se revisa su vecindario
// en una ventana de tamaño (2*haloSize+1) x (2*haloSize+1). Si se detecta al
// menos un vecino que NO es blanco (es decir, parte de la zona coloreada),
// se pinta ese píxel de negro. Los píxeles coloreados se dejan intactos.
__global__ void delineateKernel(const Pixel* d_in, Pixel* d_out, int width, int height, int haloSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Pixel p = d_in[idx];

    // Solo procesamos los píxeles que son completamente blancos.
    if (!(p.r == 255 && p.g == 255 && p.b == 255)) {
        // Si ya es un píxel coloreado, lo dejamos sin cambios.
        d_out[idx] = p;
        return;
    }

    bool adjacentColored = false;
    // Revisar vecinos en la ventana centrada en (x,y)
    for (int j = -haloSize; j <= haloSize && !adjacentColored; j++) {
        for (int i = -haloSize; i <= haloSize && !adjacentColored; i++) {
            int nx = x + i;
            int ny = y + j;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;
            int nIdx = ny * width + nx;
            Pixel np = d_in[nIdx];
            // Si se encuentra al menos un vecino que NO es blanco, es adyacente a la zona coloreada.
            if (!(np.r == 255 && np.g == 255 && np.b == 255))
                adjacentColored = true;
        }
    }

    // Si el píxel blanco está adyacente a la zona coloreada, lo marcamos como borde (negro).
    d_out[idx] = adjacentColored ? Pixel{ 0, 0, 0 } : p;
}

// -----------------------------------------------------------------------------
// Kernel: Calcula la suma ponderada de los componentes RGB de cada píxel.
// ----------------------------------------------------------------------------- 
__global__ void weightedSumKernel(Pixel* d_in, int* d_partialMax, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    if (x < width && y < height) {
        int weightedSum = static_cast<int>(d_in[idx].r * 0.50f + d_in[idx].g * 0.25f + d_in[idx].b * 0.25f);
        d_partialMax[idx] = weightedSum;
    }
}

// -----------------------------------------------------------------------------
// Kernel: Realiza una reducción paralela para encontrar el valor máximo.
// La reducción se realiza en bloques de tamaño blockDim.x * 2.
// Los resultados parciales se almacenan en d_out.
// -----------------------------------------------------------------------------
__global__ void hashKernel(int* d_in, int* d_out, int n) {
    extern __shared__ int sharedMax[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sharedMax[tid] = (idx < n) ? d_in[idx] : -2147483648;
    if (idx + blockDim.x < n) {
        sharedMax[tid] = max(sharedMax[tid], d_in[idx + blockDim.x]);
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = sharedMax[0];
    }
}

// -----------------------------------------------------------------------------
// Función auxiliar: Normaliza un valor a un rango de caracteres ASCII.
// Devuelve un valor entre 35 y 125.
// ----------------------------------------------------------------------------- 
__host__ int normalizeToASCII(int value) {
    return 35 + ((value * (125 - 35)) / 255);
}

// -----------------------------------------------------------------------------
// Kernel: Merge de imagenes.
// Combina la imagen delineada (d_del) y la imagen en blanco y negro (d_bn).
// Para cada píxel, si en la imagen delineada el píxel es blanco, toma el valor BN;
// de lo contrario conserva el valor de la imagen delineada.
// -----------------------------------------------------------------------------
__global__ void mergeKernel(const Pixel* d_del, const Pixel* d_bn, Pixel* d_out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    Pixel pd = d_del[idx];
    Pixel pbn = d_bn[idx];
    d_out[idx] = (pd.r == 255 && pd.g == 255 && pd.b == 255) ? pbn : pd;
}
//__device__ int max(int a, int b) {
//    return a ? (a >= b) : b;
//}

// -----------------------------------------------------------------------------
// Función principal para procesar la imagen.
// Para las opciones 1, 6, 21, 22, 31, 32, 33 se comporta como antes.
// Para las opciones 41, 42 y 43 (identificación + delineado + merge BN) se ejecuta el siguiente pipeline:
//   a) Se copia d_in a un buffer d_orig para preservar la imagen original.
//   b) Se llama a setColorThresholds() para fijar los thresholds.
//   c) Se lanza identifyKernel sobre d_orig, almacenando el resultado en d_out.
//   d) Se lanza delineateKernel sobre d_out con haloSize para generar el delineado (se escribe en un buffer d_del).
//   e) Se convierte d_orig a blanco y negro in-place con toGrayKernel.
//   f) Se llama a mergeKernel que fusiona d_out (delineada) y d_orig (BN), almacenando el resultado en d_merge.
//   g) Se copia d_merge al host.
// -----------------------------------------------------------------------------
int procImg(Pixel* pixels, int height, int width,
    int option, int filterDiv, unsigned int* outCount, int haloSize)
{
    cudaError_t cudaStatus;
    Pixel* d_in = nullptr;
    Pixel* d_out = nullptr;
    int totalPixels = width * height;
    if (outCount) *outCount = 0;

    // 1) Obtener información de la GPU y seleccionar dispositivo
    int devID = 0;
    cudaDeviceProp prop;
    cudaStatus = cudaGetDeviceProperties(&prop, devID);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!\n");
        return 1;
    }
    cudaStatus = cudaSetDevice(devID);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\n");
        return 1;
    }

    // 2) Calcular blockDim y gridDim iniciales
    int maxThreads = prop.maxThreadsPerBlock;
    int bCandidate = static_cast<int>(floorf(std::sqrtf((float)maxThreads)));
    if (bCandidate > prop.maxThreadsDim[0]) bCandidate = prop.maxThreadsDim[0];
    if (bCandidate > prop.maxThreadsDim[1]) bCandidate = prop.maxThreadsDim[1];
    dim3 blockDim(bCandidate, bCandidate, 1);
    int gridX = (width + blockDim.x - 1) / blockDim.x;
    int gridY = (height + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridX, gridY, 1);
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

    // 3) Reservar memoria en GPU para la imagen
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
    cudaStatus = cudaMemcpy(d_in, pixels, totalPixels * sizeof(Pixel), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy Host->Device failed!\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    // 4) Reservar contador si se requieren (para identificación)
    unsigned int* d_count = nullptr;
    if (option == 31 || option == 32 || option == 33 ||
        option == 41 || option == 42 || option == 43) {
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

    // Variable para determinar dónde está el resultado final:
    // - Si la operación es in-place, estará en d_in.
    // - Si es out-of-place, estará en d_out (o en otro buffer que asignemos).
    bool devResultIn_d_out = false;

    // 5) Lanzar el kernel según la opción:
    // Para la identificación sin halo se usan las opciones 31,32,33.
    // Para la identificación con halo (fase 04) se usan las opciones 41,42,43.
    if (option == 41 || option == 42 || option == 43) {
        // Fase 04: Identificación + delineado + fusión BN
        if (haloSize <= 0) {
            fprintf(stderr, "Para la fase de delineado, haloSize debe ser > 0.\n");
            cudaFree(d_in);
            cudaFree(d_out);
            return 1;
        }
        // a) Reservar un buffer d_orig para preservar la imagen original
        Pixel* d_orig = nullptr;
        cudaStatus = cudaMalloc((void**)&d_orig, totalPixels * sizeof(Pixel));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_orig failed!\n");
            cudaFree(d_in);
            cudaFree(d_out);
            return 1;
        }
        cudaMemcpy(d_orig, d_in, totalPixels * sizeof(Pixel), cudaMemcpyDeviceToDevice);

        // b) Fijar thresholds (se establecen internamente)
        setColorThresholds();

        // c) Identificación de color: según la opción (41: rojo, 42: verde, 43: azul)
        const int* pThresh = nullptr;
        if (option == 41)
            cudaStatus = cudaGetSymbolAddress((void**)&pThresh, c_threshRed);
        else if (option == 42)
            cudaStatus = cudaGetSymbolAddress((void**)&pThresh, c_threshGreen);
        else if (option == 43)
            cudaStatus = cudaGetSymbolAddress((void**)&pThresh, c_threshBlue);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetSymbolAddress failed: %s\n", cudaGetErrorString(cudaStatus));
            cudaFree(d_in); cudaFree(d_out); cudaFree(d_orig);
            return 1;
        }
        // Reservar un buffer d_temp para la imagen identificada
        Pixel* d_temp = nullptr;
        cudaStatus = cudaMalloc((void**)&d_temp, totalPixels * sizeof(Pixel));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_temp failed!\n");
            cudaFree(d_in); cudaFree(d_out); cudaFree(d_orig);
            return 1;
        }
        identifyKernel <<<gridDim, blockDim >>> (d_orig, d_temp, width, height, pThresh, d_count);

        // d) Delineado: aplicar delineateKernel sobre d_temp y guardar el resultado en d_del
        Pixel* d_del = nullptr;
        cudaStatus = cudaMalloc((void**)&d_del, totalPixels * sizeof(Pixel));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_del failed!\n");
            cudaFree(d_in); cudaFree(d_out); cudaFree(d_orig); cudaFree(d_temp);
            return 1;
        }
        delineateKernel <<<gridDim, blockDim >>> (d_temp, d_del, width, height, haloSize);
        cudaFree(d_temp);

        // e) Convertir d_orig a blanco y negro in-place (toGrayKernel)
        toGrayKernel <<<gridDim, blockDim >>> (d_orig, width, height);

        // f) Fusionar: mergeKernel combina la imagen delineada (d_del) y la imagen BN (d_orig)
        Pixel* d_merge = nullptr;
        cudaStatus = cudaMalloc((void**)&d_merge, totalPixels * sizeof(Pixel));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_merge failed!\n");
            cudaFree(d_in); cudaFree(d_out); cudaFree(d_orig); cudaFree(d_del);
            return 1;
        }
        mergeKernel <<<gridDim, blockDim >>> (d_del, d_orig, d_merge, width, height);
        cudaFree(d_del);
        cudaFree(d_orig);
        cudaFree(d_out);
        d_out = d_merge;
        devResultIn_d_out = true;
    }
    else {
        // Para las demás opciones, se procede como antes.
        switch (option) {
        case 1: {// Blanco y Negro (in-place)
            toGrayKernel <<<gridDim, blockDim >>> (d_in, width, height);
            devResultIn_d_out = false;
        }
            break;
        case 6: {// Invertir colores (in-place)
            invertColorsKernel <<<gridDim, blockDim >>> (d_in, width, height);
            devResultIn_d_out = false;
        }
            break;
        case 21: { // Pixelar en color (out-of-place)
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
            size_t sizeData = nThreads * sizeof(Pixel);
            size_t sizeR = nThreads * sizeof(int);
            size_t sizeG = nThreads * sizeof(int);
            size_t sizeB = nThreads * sizeof(int);
            size_t sizeCount = nThreads * sizeof(int);
            size_t totalNeeded = sizeData + sizeR + sizeG + sizeB + sizeCount;
            pixelateKernel <<<gridDim, blockDim, totalNeeded >>> (d_in, d_out, width, height);
            devResultIn_d_out = true;
        }
               break;
        case 22: { // Pixelar BN (BN in-place, luego pixelado out-of-place)
            toGrayKernel <<<gridDim, blockDim >>> (d_in, width, height);
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
            size_t sizeData = nThreads * sizeof(Pixel);
            size_t sizeR = nThreads * sizeof(int);
            size_t sizeG = nThreads * sizeof(int);
            size_t sizeB = nThreads * sizeof(int);
            size_t sizeCount = nThreads * sizeof(int);
            size_t totalNeeded = sizeData + sizeR + sizeG + sizeB + sizeCount;
            pixelateKernel <<<gridDim, blockDim, totalNeeded >>> (d_in, d_out, width, height);
            devResultIn_d_out = true;
        }
               break;
        case 31: { // Identificar Rojo (sin halo) => out-of-place
            setColorThresholds();
            const int* pRed = nullptr;
            cudaStatus = cudaGetSymbolAddress((void**)&pRed, c_threshRed);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaGetSymbolAddress (c_threshRed) failed: %s\n", cudaGetErrorString(cudaStatus));
                cudaFree(d_in);
                cudaFree(d_out);
                return 1;
            }
            identifyKernel <<<gridDim, blockDim >>> (d_in, d_out, width, height, pRed, d_count);
            devResultIn_d_out = true;
        }
               break;
        case 32: { // Identificar Verde (sin halo) => out-of-place
            setColorThresholds();
            const int* pGreen = nullptr;
            cudaStatus = cudaGetSymbolAddress((void**)&pGreen, c_threshGreen);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaGetSymbolAddress (c_threshGreen) failed: %s\n", cudaGetErrorString(cudaStatus));
                cudaFree(d_in);
                cudaFree(d_out);
                return 1;
            }
            identifyKernel <<<gridDim, blockDim >>> (d_in, d_out, width, height, pGreen, d_count);
            devResultIn_d_out = true;
        }
               break;
        case 33: { // Identificar Azul (sin halo) => out-of-place
            setColorThresholds();
            const int* pBlue = nullptr;
            cudaStatus = cudaGetSymbolAddress((void**)&pBlue, c_threshBlue);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaGetSymbolAddress (c_threshBlue) failed: %s\n", cudaGetErrorString(cudaStatus));
                cudaFree(d_in);
                cudaFree(d_out);
                return 1;
            }
            identifyKernel <<<gridDim, blockDim >>> (d_in, d_out, width, height, pBlue, d_count);
            devResultIn_d_out = true;
        }
               break;
		case 5:{ // Pseudo-hash
            int* d_partialMax = nullptr;
            int* d_max = nullptr;

            cudaMalloc(&d_partialMax, totalPixels * sizeof(int));

            weightedSumKernel <<<gridDim, blockDim >>> (d_in, d_partialMax, width, height);
            cudaDeviceSynchronize();

            int curSize = totalPixels;
            bool firstReduction = true;
            while (curSize > 15) {
                int threadsPerBlock = firstReduction ? 512 : 15;
                int newBlocks = (curSize + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
                cudaMalloc(&d_max, newBlocks * sizeof(int));
                hashKernel <<<newBlocks, threadsPerBlock, threadsPerBlock * sizeof(int) >> > (d_partialMax, d_max, curSize);
                cudaDeviceSynchronize();
                cudaFree(d_partialMax);
                d_partialMax = d_max;
                curSize = newBlocks;
                firstReduction = false;
            }

            int h_max[15];
            cudaMemcpy(h_max, d_partialMax, curSize * sizeof(int), cudaMemcpyDeviceToHost);

            printf("Unnormalized values: ");
            for (int i = 0; i < curSize; i++) {
                printf("%d ", h_max[i]);
            }
            printf("\n");

            printf("Normalized ASCII values: ");
            for (int i = 0; i < curSize; i++) {
                h_max[i] = normalizeToASCII(h_max[i]);
                printf("%d ", static_cast<char>(h_max[i]));
            }
            printf("");

            cudaFree(d_in);
            cudaFree(d_partialMax);
            return 0;
		}
        default:
            fprintf(stderr, "Opcion %d no reconocida.\n", option);
            cudaFree(d_in);
            cudaFree(d_out);
            return 1;
        }
    }

    // 8) Verificar errores de lanzamiento del kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    // 9) Sincronizar
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error %d\n", cudaStatus);
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    // 10) Copiar el resultado de vuelta al host
    Pixel* d_result = devResultIn_d_out ? d_out : d_in;
    cudaStatus = cudaMemcpy(pixels, d_result, totalPixels * sizeof(Pixel), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D->H failed!\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    // 11) Si se usaron contadores (para identificación), copiarlos a outCount
    if (option == 31 || option == 32 || option == 33 ||
        option == 41 || option == 42 || option == 43) {
        unsigned int hCount = 0;
        cudaMemcpy(&hCount, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (outCount) *outCount = hCount;
        cudaFree(d_count);
    }

    // 12) Liberar memoria en GPU
    cudaFree(d_in);
    cudaFree(d_out);

    // 13) (Opcional) Reset del dispositivo (comentado para permitir múltiples llamadas)
    // cudaDeviceReset();

    return 0; // éxito
}
