
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <math.h>
#include <time.h>

// OpenGL Graphics includes
#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#define GRIDSIZE 512

#define SCREEN_X GRIDSIZE
#define SCREEN_Y GRIDSIZE
#define FPS_UPDATE 500
#define TITLE "Lenia"

#define CPU_MODE 1
#define GPU_MODE 2
#define CPU_EXT_MODE 3
#define GPU_EXT_MODE 4

#define width_grid GRIDSIZE
#define height_grid GRIDSIZE

int block_dim_x = 16;
int block_dim_y = 16;
int grid_dim_x;
int grid_dim_y;

float4* d_grid1, * d_grid2;
bool tab_1_used = true;
bool show_kernel = false;

GLuint imageTex;
GLuint imageBuffer;

GLuint glBuffer;
GLuint glTex;
struct cudaGraphicsResource* cuBuffer;

float* debug;

/* Globals */
float scale = 0.003f;
float mx = 0.f;
float my = 0.f;
int mode = GPU_MODE;
int frame = 0;
int timebase = 0;

// Lenia parameters

#define R 12
#define T 10
#define mu 0.15f
#define sigma 0.017f
#define alpha 4
#define B 3 // rank for the pics

// Kernel struct
struct Noyau {
    int id;
    int range;
    int source_channel;
    int dest_channel;
    float h;
    float beta[B];
    float k_s_norm;
    float* n_kernel;
};

Noyau* kernel; // basic version
Noyau* kernels[3]; // expanded version
Noyau kernel1; Noyau kernel0; Noyau kernel2;

float4* lenia_pixels;
float4* lenia_pixels2;
int size;

float* n_kernel;
float k_s_norm;
float beta[B];
__constant__ float cm_beta[B];

float* ks_tab;
float* ks_tab1; float* ks_tab2;
float* d_ks_tab;
float* d_ks_tab1; float* d_ks_tab2;


#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors
(cudaError err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString(err));
        system("pause");
        exit(1);
    }
}

void initOpenGlCUDA()
{
    // create a buffer
    glGenBuffers(1, &glBuffer); 
    // make it the active buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, glBuffer);
    // allocate memory, but dont copy data (NULL)
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_STREAM_DRAW);

    glEnable(GL_TEXTURE_2D); // Enable texturing
    glGenTextures(1, &glTex); // Generate a texture ID
    glBindTexture(GL_TEXTURE_2D, glTex); // Set as the current texture
    // Allocate the texture memory.
    // The last parameter is NULL:
    // we only want to allocate memory, not initialize it
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCREEN_X, SCREEN_Y, 0, GL_RGBA, GL_FLOAT, NULL);

    // Must set the filter mode:
    // GL_LINEAR enables interpolation when scaling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //cudaGLSetGLDevice(0); // explicitly set device 0
    cudaGraphicsGLRegisterBuffer(&cuBuffer, glBuffer, cudaGraphicsMapFlagsWriteDiscard);
    // cudaGraphicsMapFlagsWriteDiscard:
    // CUDA will only write and will not read from this resource
}

float4* zeroPixels() {
    float4* p = (float4*)malloc(size);
    for (int i = 0; i < width_grid * height_grid;i++) {
        p[i].x = 0.0f;
        p[i].y = 0.0f;
        p[i].z = 0.0f;
        p[i].w = 1.0f;
    }
    return p;
}

float4* randomPixels(int nbChannels) {
    float4* p = (float4*)malloc(size);
    for (int i = 0; i < width_grid * height_grid; i++) {
        if (i % (10 * width_grid) == 0) srand(i);
        float random_value = (float)rand() / RAND_MAX; // Generate a random float between 0 and 1
        float random_value2 = (float)rand() / RAND_MAX;
        float random_value3 = (float)rand() / RAND_MAX;
        float test_alive = (float)rand() / RAND_MAX;

        if (test_alive < 0.77f) {
            random_value = 0.0f;
            random_value2 = 0.0f;
            random_value3 = 0.0f;
        }

        p[i].x = (nbChannels > 1) ? random_value3 : 0.0f;
        p[i].y = random_value;
        p[i].z = (nbChannels == 3) ? random_value2 : 0.0f;
        p[i].w = 1.0f;
    }
    return p;
}

float4* randomPixelsCenter(int nbChannels) {
    float4* p = (float4*)malloc(size);
    for (int i = 3 * height_grid / 8; i < 5 * height_grid / 8; i++) {
        for (int j = 3 * width_grid / 8; j < 5 * width_grid / 8; j++) {
            int index = i * width_grid + j;

            if (index % (10 * width_grid) == 0) srand(index);
            float random_value = (float)rand() / RAND_MAX; // Generate a random float between 0 and 1
            float random_value2 = (float)rand() / RAND_MAX;
            float random_value3 = (float)rand() / RAND_MAX;
            p[index].x = (nbChannels > 1) ? random_value3 : 0.0f;
            p[index].y = random_value;
            p[index].z = (nbChannels == 3) ? random_value2 : 0.0f;
            p[index].w = 1.0f;
        }

    }
    return p;
}

float4* glider()
{
    float4* p = zeroPixels();
    p[0 * width_grid + 1].y = 1.0f;
    p[1 * width_grid + 2].y = 1.0f;
    p[2 * width_grid + 0].y = 1.0f;
    p[2 * width_grid + 1].y = 1.0f;
    p[2 * width_grid + 2].y = 1.0f;
    return p;
}

// Free Noyau resources
void freeNoyau(Noyau* n) {
    free(n->n_kernel);
    free(n);
}

void cleanCPU()
{
    free(lenia_pixels);
    free(lenia_pixels2);
    freeNoyau(kernel);
    free(ks_tab);
}

void cleanExtendedCPU()
{
    free(lenia_pixels);
    free(lenia_pixels2);

    for (Noyau* k : kernels) {
        freeNoyau(k);
    }

    free(ks_tab); free(ks_tab1); free(ks_tab2);
}

void cleanGPU()
{
    free(lenia_pixels);
    freeNoyau(kernel);
    free(ks_tab);

    cudaGraphicsUnregisterResource(cuBuffer);
    glDeleteTextures(1, &glTex);
    glDeleteBuffers(1, &glBuffer);

    cudaFree(d_grid1);
    cudaFree(d_grid2);
    cudaFree(d_ks_tab);
}

void cleanExtendedGPU()
{
    free(lenia_pixels);
    for (Noyau* k : kernels) {
        freeNoyau(k);
    }
    free(ks_tab); free(ks_tab1); free(ks_tab2);

    cudaGraphicsUnregisterResource(cuBuffer);
    glDeleteTextures(1, &glTex);
    glDeleteBuffers(1, &glBuffer);

    cudaFree(d_grid1); cudaFree(d_grid2);
    cudaFree(d_ks_tab); cudaFree(d_ks_tab1); cudaFree(d_ks_tab2);
}


/*
    exponential core function
    pre-condition : r between 0 and 1
    post-condition : return value between 0 and 1
*/
__host__ __device__ float kernel_core_exp(float r)
{
    float k = (4.0f * r * (1.0f - r));
    float k_core = exp(alpha * (1.0f - 1.0f / k));
    return k_core;
}

__host__ __device__ float growth_function_exp(float u)
{
    float dividende = (2.0f * sigma * sigma);
    return 2.0f * exp(-((u - mu) * (u - mu)) / dividende) - 1.0f;
}

// Like the Game of Life

float kernel_core_step(float r)
{
    float k = 0.0f;
    float q = 0.25f;
    if (q <= r && r <= (1 - q)) {
        k = 1.0f;
    }
    else if (r <= q) {
        k = 0.5f;
    }
    return k;
}

float growth_function_step(float u) {
    float l = u - mu;
    if (l < 0.0f) {
        l = -l;
    }

    float d = -1.0f;
    if (l <= sigma) {
        d = 1.0f;
    }
    return d;
}

__host__ __device__ int emod(int a, int b)
{
    int res = a;
    if (a < 0)
    {
        res = b + a;
    }
    else if (a > b)
    {
        res = a - b;
    }
    return res;
}


/*
    kernel shell
    pre-conditions : r between 0 and 1 ; beta in [0;1] dim B
    post-condition : return value between 0 and 1
*/
__host__ __device__ float kernel_shell(float n, float beta[])
{
    int index = emod(floor((float)B * n), B);
    float peak_height = beta[index];
    float u = fmod(float(B * n), 1.0f);
    return peak_height * kernel_core_exp(u);
}

__host__ double toric_distance(int x1, int y1, int x2, int y2, int width, int height) {
    int dx = abs(x1 - x2);
    int dy = abs(y1 - y2);

    if (dx > width / 2) {
        dx = width - dx;
    }
    if (dy > height / 2) {
        dy = height - dy;
    }

    return sqrt(dx * dx + dy * dy);
}
/*
    Normalization of the kernel
    pre-condition : n in the neighbourhood, at the indexes i and j
    post-condition :
*/
__host__ float normalized_kernel(int x, int y, float beta[])
{
    int center_x = width_grid / 2;
    int center_y = height_grid / 2;

    float norm_n = toric_distance(center_x, center_y, x, y, width_grid, height_grid) / R;
    float Ks_val = 0.0f;
    if (toric_distance(center_x, center_y, x, y, width_grid, height_grid) <= R) {
        Ks_val = kernel_shell(norm_n, beta);
    }

    return Ks_val;
}

void calculateKsNorm(Noyau* k)
{
    float acc = 0;
    for (int i = -k->range; i <= k->range; i++) {
        for (int j = -k->range; j <= k->range; j++) {
            float r = (float)sqrt(i * i + j * j) / (float)k->range;
            if (r <= 1.0f) {  // within the radius
                acc += kernel_shell(r, beta);
            }
        }
    }
    k->k_s_norm = acc;
}

void init_kernel(Noyau* k) {

    k->n_kernel = (float*)malloc(width_grid * height_grid * sizeof(float));

    for (int x = 0; x < height_grid; x++) {
        for (int y = 0; y < width_grid; y++) {
            float val = normalized_kernel(x, y, beta);
            k->n_kernel[x * width_grid + y] = val;
        }
    }
}

__host__ float* init_ks_tab(float beta[])
{
    int w = (2 * R + 1);
    float* ks_t = (float*)malloc(w * w * sizeof(float));

    for (int i = -R; i <= R; i++)
    {
        for (int j = -R; j <= R; j++)
        {
            int ni = (i + w) % w;
            int nj = (j + w) % w;
            ks_t[ni * w + nj] = 0;
            float r = sqrtf(i * i + j * j) / R;
            if (r <= 1.0f) {  // within the radius
                ks_t[ni * w + nj] = kernel_shell(r, beta);
            }
        }
    }
    return ks_t;
}

// Initialize Noyau
void initNoyau(Noyau* n, int id, int r, int src, int dest, float h, float b0, float b1, float b2) {
    n->id = id;
    n->range = r;
    n->source_channel = src;
    n->dest_channel = dest;
    n->h = h;
    if (B == 1) {
        n->beta[0] = b0;
    }
    else {
        n->beta[0] = b0;
        n->beta[1] = b1;
        n->beta[2] = b2;
    }

    calculateKsNorm(n);
    init_kernel(n);
}


void initCPU()
{
    tab_1_used = true;

    beta[0] = 1.0f;
    beta[1] = 1.0f / 3.0f;
    beta[2] = 7.0f / 12.0f;

    lenia_pixels = randomPixelsCenter(1);
    lenia_pixels2 = zeroPixels();

    kernel = (Noyau*)malloc(sizeof(Noyau));
    initNoyau(kernel, 0, R, 0, 0, 1, beta[0], beta[1], beta[2]);

    ks_tab = init_ks_tab(beta);

    k_s_norm = kernel->k_s_norm;
}

void initExtendedCPU()
{
    tab_1_used = true;

    lenia_pixels = randomPixels(3);
    lenia_pixels2 = zeroPixels();

    Noyau* k0 = (Noyau*)malloc(sizeof(Noyau)); initNoyau(k0, 0, R, 0, 1, 1, 1.0f, 1.0f / 3.0f, 7.0f / 12.0f); kernels[0] = k0;
    Noyau* k1 = (Noyau*)malloc(sizeof(Noyau)); initNoyau(k1, 1, R, 1, 2, 1, 7.0f / 12.0f, 1.0f, 1.0f / 3.0f); kernels[1] = k1;
    Noyau* k2 = (Noyau*)malloc(sizeof(Noyau)); initNoyau(k2, 2, R, 2, 0, 1, 1.0f / 3.0f, 7.0f / 12.0f, 1.0f); kernels[2] = k2;

    ks_tab = init_ks_tab(k0->beta);
    ks_tab1 = init_ks_tab(k1->beta);
    ks_tab2 = init_ks_tab(k2->beta);
}

void initGPU()
{
    tab_1_used = true;

    beta[0] = 1.0f;
    beta[1] = 1.0f / 3.0f;
    beta[2] = 7.0f / 12.0f;
    cudaMemcpyToSymbol(cm_beta, beta, B * sizeof(float));

    initOpenGlCUDA();

    checkCudaErrors(cudaMalloc((void**)&d_grid1, size));
    checkCudaErrors(cudaMalloc((void**)&d_grid2, size));

    lenia_pixels = randomPixelsCenter(1);
    checkCudaErrors(cudaMemcpy(d_grid1, lenia_pixels, size, cudaMemcpyHostToDevice));

    kernel = (Noyau*)malloc(sizeof(Noyau));
    initNoyau(kernel, 0, R, 0, 0, 1, beta[0], beta[1], beta[2]);

    k_s_norm = kernel->k_s_norm;

    ks_tab = init_ks_tab(beta);
    checkCudaErrors(cudaMalloc((void**)&d_ks_tab, (2 * R + 1) * (2 * R + 1) * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_ks_tab, ks_tab, (2 * R + 1) * (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice));
}

void initExtendedGPU()
{
    tab_1_used = true;

    initOpenGlCUDA();

    checkCudaErrors(cudaMalloc((void**)&d_grid1, size));
    checkCudaErrors(cudaMalloc((void**)&d_grid2, size));

    lenia_pixels = randomPixels(3);
    checkCudaErrors(cudaMemcpy(d_grid1, lenia_pixels, size, cudaMemcpyHostToDevice));

    Noyau* k0 = (Noyau*)malloc(sizeof(Noyau)); initNoyau(k0, 0, R, 0, 1, 1, 1.0f, 1.0f / 3.0f, 7.0f / 12.0f); kernels[0] = k0; kernel0 = *k0;
    Noyau* k1 = (Noyau*)malloc(sizeof(Noyau)); initNoyau(k1, 1, R, 1, 2, 1, 7.0f / 12.0f, 1.0f, 1.0f / 3.0f); kernels[1] = k1; kernel1 = *k1;
    Noyau* k2 = (Noyau*)malloc(sizeof(Noyau)); initNoyau(k2, 2, R, 2, 0, 1, 1.0f / 3.0f, 7.0f / 12.0f, 1.0f); kernels[2] = k2; kernel2 = *k2;

    ks_tab = init_ks_tab(k0->beta);
    checkCudaErrors(cudaMalloc((void**)&d_ks_tab, (2 * R + 1) * (2 * R + 1) * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_ks_tab, ks_tab, (2 * R + 1) * (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice));
    ks_tab1 = init_ks_tab(k1->beta);
    checkCudaErrors(cudaMalloc((void**)&d_ks_tab1, (2 * R + 1) * (2 * R + 1) * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_ks_tab1, ks_tab1, (2 * R + 1) * (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice));
    ks_tab2 = init_ks_tab(k2->beta);
    checkCudaErrors(cudaMalloc((void**)&d_ks_tab2, (2 * R + 1) * (2 * R + 1) * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_ks_tab2, ks_tab2, (2 * R + 1) * (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice));
}


/*
    Potential distribution U_t(x) - Local rule
    Pre-condition : in the grid at indexes x and y
    Post-condition : return value between 0 and 1
*/
__host__ __device__ float potential_distribution(int x, int y, float4 d_grid_old[], float k_s_norm, float beta[], int width, int height, float ks_tab[]) {

    int w = (2 * R + 1);
    float sum = 0.0f;
    for (int i = -R; i <= R; i++)
    {
        for (int j = -R; j <= R; j++)
        {
            int ni = (x + i + height) % height;
            int nj = (y + j + width) % width;
            int wi = (i + w) % w;
            int wj = (j + w) % w;
            float r = sqrtf(i * i + j * j) / R;
            if (r <= 1.0f) {  // within the radius
                sum += ks_tab[wi * w + wj] * d_grid_old[ni * width + nj].y;
            }
        }
    }

    return sum / k_s_norm;
}

__host__ __device__ void computeBasicLenia(int i, int j, float4* p1, float4* p2, float k_s_norm, float* beta, float* ks_tab) {
    int index = i * width_grid + j;
    float c_t = p1[index].y; // field C at time step t
    float c_tdt; // field C at time step t + delta(t) ; (delta (t) = 1/T)

    // 1st step : convolution operation, multiplication with the kernel
    float u_t = potential_distribution(i, j, p1, k_s_norm, beta, width_grid, height_grid, ks_tab);
    // 2nd step : growth mapping
    float g_t = growth_function_exp(u_t);
    // 3rd step : add the growth to the existing value
    float dt = 1.0f / float(T);
    c_tdt = c_t + dt * g_t;
    // 4th step : clip the result to be in range from 0 to 1
    if (c_tdt < 0.0f) c_tdt = 0.0f;
    if (c_tdt > 1.0f) c_tdt = 1.0f;

    // assign the value
    p2[index].x = 0.0f;
    p2[index].y = c_tdt;
    p2[index].z = 0.0f;
    p2[index].w = 1.0f;
}

__host__ void lenia_basic_CPU(float4* p1, float4* p2)
{
    // for each pixel
    for (int i = 0; i < height_grid; i++) {
        for (int j = 0; j < width_grid; j++) {
            computeBasicLenia(i, j, p1, p2, k_s_norm, beta, ks_tab);
        }
    }
}


__global__ void lenia_gpu(float4* p1, float4* p2, float4* cuPixels, float k_s_norm, float* d_ks_tab)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int index = i * width_grid + j;

    if (i < height_grid && j < width_grid) {
        computeBasicLenia(i, j, p1, p2, k_s_norm, cm_beta, d_ks_tab);
        cuPixels[index] = p2[index];
    }
}


void lenia_basic_GPU()
{
    dim3 dimBlock(block_dim_x, block_dim_y);
    dim3 dimGrid(grid_dim_x, grid_dim_y);

    cudaError_t err;

    //OpenGL interoperability
    cudaGraphicsMapResources(1, &cuBuffer, 0);
    float4* cuPixels;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&cuPixels, &num_bytes, cuBuffer);

    // do treatments
    if (tab_1_used) {
        lenia_gpu << < dimGrid, dimBlock >> > (d_grid1, d_grid2, cuPixels, k_s_norm, d_ks_tab);
    }
    else {
        lenia_gpu << < dimGrid, dimBlock >> > (d_grid2, d_grid1, cuPixels, k_s_norm, d_ks_tab);
    }

    cudaGraphicsUnmapResources(1, &cuBuffer);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
}



// ----------------------- EXPANDED VERSION ----------------------------------------------------------




/*
    Potential distribution U_t(x) - Local rule
    Pre-condition : in the grid at indexes x and y
    Post-condition : return value between 0 and 1
*/
__host__ __device__ float potential_distribution_ext(Noyau* k, int x, int y, float4 d_grid_old[], int width, int height, float ks_tab[], float ks_tab1[], float ks_tab2[]) {

    int w = (2 * R + 1);
    float sum = 0.0f;
    for (int i = -R; i <= R; i++)
    {
        for (int j = -R; j <= R; j++)
        {
            int ni = (x + i + height) % height;
            int nj = (y + j + width) % width;
            int wi = (i + w) % w;
            int wj = (j + w) % w;

            float r = sqrtf(i * i + j * j) / R;
            if (r <= 1.0f) {  // within the radius

                float grid_value;
                switch (k->source_channel)
                {
                case 0:
                    grid_value = d_grid_old[ni * width + nj].x;
                    break;
                case 1:
                    grid_value = d_grid_old[ni * width + nj].y;
                    break;
                default:
                    grid_value = d_grid_old[ni * width + nj].z;
                    break;
                }

                float ks_value;
                switch (k->id){
                case 0:
                    ks_value = ks_tab[wi * w + wj];
                    break;
                case 1:
                    ks_value = ks_tab1[wi * w + wj];
                    break;
                default:
                    ks_value = ks_tab2[wi * w + wj];
                    break;
                }

                sum += ks_value * grid_value;
            }
        }
    }
    return sum / k->k_s_norm;
}


__host__ void computeExtendedLenia(int i, int j, float4* p1, float4* p2) {


    int index = i * width_grid + j;
    float c_tdt_x, c_tdt_y, c_tdt_z; // field C at time step t + delta(t) ; (delta (t) = 1/T)

    float h_tot = kernels[0]->h + kernels[1]->h + kernels[2]->h;

    // for each kernel K(k) : 
    for (Noyau* k : kernels)
    {
        // 1st step : convolution operation, multiplication with the kernel k with source A
        float u_t = potential_distribution_ext(k, i, j, p1, width_grid, height_grid, ks_tab, ks_tab1, ks_tab2);
        // 2nd step : growth mapping
        float g_t = growth_function_exp(u_t);
        // 3rd step : add the growth to the existing value of the dest channel
        float dt = 1.0f / float(T);
        float weighted_sum = (k->h / h_tot);

        switch (k->dest_channel)
        {
        case 0:
            c_tdt_x = p1[index].x + dt * weighted_sum * g_t;
            break;
        case 1:
            c_tdt_y = p1[index].y + dt * weighted_sum * g_t;
            break;
        default:
            c_tdt_z = p1[index].z + dt * weighted_sum * g_t;
            break;
        }
    }


    // 4th step : clip the result to be in range from 0 to 1
    if (c_tdt_x < 0.0f) c_tdt_x = 0.0f;
    if (c_tdt_x > 1.0f) c_tdt_x = 1.0f;
    if (c_tdt_y < 0.0f) c_tdt_y = 0.0f;
    if (c_tdt_y > 1.0f) c_tdt_y = 1.0f;
    if (c_tdt_z < 0.0f) c_tdt_z = 0.0f;
    if (c_tdt_z > 1.0f) c_tdt_z = 1.0f;

    // assign the value
    p2[index].x = c_tdt_x;
    p2[index].y = c_tdt_y;
    p2[index].z = c_tdt_z;
    p2[index].w = 1.0f;
}

__device__ void computeExtendedLeniaGpu(int i, int j, float4* p1, float4* p2, Noyau kernel0, Noyau kernel1, Noyau kernel2, float* d_ks_tab, float* d_ks_tab1, float* d_ks_tab2) {


    int index = i * width_grid + j;
    float c_tdt_x, c_tdt_y, c_tdt_z; // field C at time step t + delta(t) ; (delta (t) = 1/T)

    float h_tot = kernel0.h + kernel1.h + kernel2.h;

    // for each kernel K(k) : 
    for (int l=0; l<3 ;l++)
    {
        Noyau k;
        switch (l)
        {
        case 0:
            k = kernel0;
            break;
        case 1:
            k = kernel1;
            break;
        default:
            k = kernel2;
            break;
        }

        // 1st step : convolution operation, multiplication with the kernel k with source A
        float u_t = potential_distribution_ext(&k, i, j, p1, width_grid, height_grid, d_ks_tab, d_ks_tab1, d_ks_tab2);
        // 2nd step : growth mapping
        float g_t = growth_function_exp(u_t);
        // 3rd step : add the growth to the existing value of the dest channel
        float dt = 1.0f / float(T);
        float weighted_sum = (k.h / h_tot);

        switch (k.dest_channel)
        {
        case 0:
            c_tdt_x = p1[index].x + dt * weighted_sum * g_t;
            break;
        case 1:
            c_tdt_y = p1[index].y + dt * weighted_sum * g_t;
            break;
        default:
            c_tdt_z = p1[index].z + dt * weighted_sum * g_t;
            break;
        }
    }


    // 4th step : clip the result to be in range from 0 to 1
    if (c_tdt_x < 0.0f) c_tdt_x = 0.0f;
    if (c_tdt_x > 1.0f) c_tdt_x = 1.0f;
    if (c_tdt_y < 0.0f) c_tdt_y = 0.0f;
    if (c_tdt_y > 1.0f) c_tdt_y = 1.0f;
    if (c_tdt_z < 0.0f) c_tdt_z = 0.0f;
    if (c_tdt_z > 1.0f) c_tdt_z = 1.0f;

    // assign the value
    p2[index].x = c_tdt_x;
    p2[index].y = c_tdt_y;
    p2[index].z = c_tdt_z;
    p2[index].w = 1.0f;
}


__host__ void lenia_extended_CPU(float4* p1, float4* p2)
{
    // for each pixel
    for (int i = 0; i < height_grid; i++) {
        for (int j = 0; j < width_grid; j++) {
            computeExtendedLenia(i, j, p1, p2);
        }
    }
}

__global__ void lenia_ext_gpu(float4* p1, float4* p2, float4* cuPixels, Noyau kernel0, Noyau kernel1, Noyau kernel2, float* d_ks_tab, float* d_ks_tab1, float* d_ks_tab2)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int index = i * width_grid + j;

    if (i < height_grid && j < width_grid) {
        computeExtendedLeniaGpu(i, j, p1, p2, kernel0, kernel1, kernel2, d_ks_tab, d_ks_tab1, d_ks_tab2);
        cuPixels[index] = p2[index];
    }
}

void lenia_extended_GPU()
{
    dim3 dimBlock(block_dim_x, block_dim_y);
    dim3 dimGrid(grid_dim_x, grid_dim_y);

    cudaError_t err;

    //OpenGL interoperability
    cudaGraphicsMapResources(1, &cuBuffer, 0);
    float4* cuPixels;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&cuPixels, &num_bytes, cuBuffer);

    // do treatments
    if (tab_1_used) {
        lenia_ext_gpu << < dimGrid, dimBlock >> > (d_grid1, d_grid2, cuPixels, kernel0, kernel1, kernel2, d_ks_tab, d_ks_tab1, d_ks_tab2);
    }
    else {
        lenia_ext_gpu << < dimGrid, dimBlock >> > (d_grid2, d_grid1, cuPixels, kernel0, kernel1, kernel2, d_ks_tab, d_ks_tab1, d_ks_tab2);
    }

    cudaGraphicsUnmapResources(1, &cuBuffer);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
}



// --------------------------------------------------- RENDERING --------------------------------------------------------------------------


void calculate() {
    frame++;
    int timecur = glutGet(GLUT_ELAPSED_TIME);

    if (timecur - timebase > FPS_UPDATE) {
        char t[200];
        char* m = "";
        switch (mode)
        {
        case CPU_MODE: m = "CPU mode"; break;
        case GPU_MODE: m = "GPU mode"; break;
        case CPU_EXT_MODE: m = "CPU extended mode"; break;
        case GPU_EXT_MODE: m = "GPU extended mode"; break;
        }
        sprintf(t, "%s:  %s, %.2f FPS", TITLE, m, frame * 1000 / (float)(timecur - timebase));
        glutSetWindowTitle(t);
        timebase = timecur;
        frame = 0;
    }

    switch (mode)
    {
    case CPU_MODE:
        if (tab_1_used) lenia_basic_CPU(lenia_pixels, lenia_pixels2);
        else  lenia_basic_CPU(lenia_pixels2, lenia_pixels);
        break;
    case CPU_EXT_MODE:
        if (tab_1_used) lenia_extended_CPU(lenia_pixels, lenia_pixels2);
        else  lenia_extended_CPU(lenia_pixels2, lenia_pixels);
        break;
    case GPU_MODE:
        lenia_basic_GPU();
        break;
    case GPU_EXT_MODE:
        lenia_extended_GPU();
        break;
    }
}

void idle()
{
    glutPostRedisplay();
}


/**
    Draw the pixels so one cell is more than one pixel
    Pre-condition : CPU mode only
*/

void draw_pixels_zoomed()
{
    // Calculate the size of each grid cell on the screen
    int cell_size_x = SCREEN_X / width_grid;
    int cell_size_y = SCREEN_Y / height_grid;

    for (int i = 0; i < height_grid; i++)
    {
        for (int j = 0; j < width_grid; j++)
        {
            // Get the color of the current pixel in the grid
            float4 color = lenia_pixels[i * width_grid + j];

            // Draw a rectangle representing the grid cell
            glBegin(GL_QUADS);
            glColor4f(color.x, color.y, color.z, color.w);
            glVertex2i(j * cell_size_x, i * cell_size_y); // Top-left corner
            glVertex2i((j + 1) * cell_size_x, i * cell_size_y); // Top-right corner
            glVertex2i((j + 1) * cell_size_x, (i + 1) * cell_size_y); // Bottom-right corner
            glVertex2i(j * cell_size_x, (i + 1) * cell_size_y); // Bottom-left corner
            glEnd();
        }
    }
}

void draw_kernel()
{
    // Calculate the size of each grid cell on the screen
    int cell_size_x = SCREEN_X / width_grid;
    int cell_size_y = SCREEN_Y / height_grid;

    for (int i = 0; i < height_grid; i++)
    {
        for (int j = 0; j < width_grid; j++)
        {
            float color_val = kernel->n_kernel[i * width_grid + j];
            // Draw a rectangle representing the grid cell
            glBegin(GL_QUADS);
            glColor4f(0.0f, 0.0f, color_val, 1.0f);
            glVertex2i(j * cell_size_x, i * cell_size_y); // Top-left corner
            glVertex2i((j + 1) * cell_size_x, i * cell_size_y); // Top-right corner
            glVertex2i((j + 1) * cell_size_x, (i + 1) * cell_size_y); // Bottom-right corner
            glVertex2i(j * cell_size_x, (i + 1) * cell_size_y); // Bottom-left corner
            glEnd();
        }
    }
}


void render()
{
    calculate();

    if (show_kernel) {
        draw_kernel();
    }
    else {
        // mode CPU
        if (mode == CPU_MODE || mode == CPU_EXT_MODE) {
            if (SCREEN_X == width_grid && SCREEN_Y == height_grid) {
                glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, tab_1_used ? lenia_pixels2 : lenia_pixels);
            }
            else {
                draw_pixels_zoomed();
            }
        }
        // mode GPU
        else {

            // Select the appropriate buffer
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glBuffer);
            // Select the appropriate texture
            glBindTexture(GL_TEXTURE_2D, glTex);
            // Make a texture from the buffer
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, NULL);

            glBegin(GL_QUADS);
            glTexCoord2f(0, 1.0f);
            glVertex3f(0, 0, 0);
            glTexCoord2f(0, 0);
            glVertex3f(0, SCREEN_Y, 0);
            glTexCoord2f(1.0f, 0);
            glVertex3f(SCREEN_X, SCREEN_Y, 0);
            glTexCoord2f(1.0f, 1.0f);
            glVertex3f(SCREEN_X, 0, 0);
            glEnd();
        }
    }

    tab_1_used = !tab_1_used;

    glutSwapBuffers();
}

void clean()
{
    switch (mode)
    {
    case CPU_MODE: cleanCPU(); break;
    case GPU_MODE: cleanGPU(); break;
    case CPU_EXT_MODE: cleanExtendedCPU(); break;
    case GPU_EXT_MODE: cleanExtendedGPU(); break;
    }

}

void init()
{

    // Intializes random number generator
    time_t t;
    srand((unsigned)time(&t));
    //srand(11);

    tab_1_used = true;
    size = width_grid * height_grid * sizeof(float4);
    grid_dim_x = (width_grid + block_dim_x - 1) / block_dim_x;
    grid_dim_y = (height_grid + block_dim_y - 1) / block_dim_y;

    switch (mode)
    {
    case CPU_MODE: initCPU(); break;
    case GPU_MODE: initGPU(); break;
    case CPU_EXT_MODE: initExtendedCPU(); break;
    case GPU_EXT_MODE: initExtendedGPU(); break;
    }
}

void toggleMode(int m)
{
    clean();
    mode = m;
    init();
}

void mouse(int button, int state, int x, int y)
{
    if (button <= 2)
    {
        mx = (float)(scale * (x - SCREEN_X / 2));
        my = -(float)(scale * (y - SCREEN_Y / 2));
    }
    // Wheel reports as button 3 (scroll up) and button 4 (scroll down)
    if (button == 3) scale /= 1.05f;
    else if (button == 4) scale *= 1.05f;
}

void mouseMotion(int x, int y)
{
    mx = (float)(scale * (x - SCREEN_X / 2));
    my = -(float)(scale * (y - SCREEN_Y / 2));
}

void processNormalKeys(unsigned char key, int x, int y) {

    if (key == 27) { clean(); exit(0); }
    else if (key == '1') toggleMode(CPU_MODE);
    else if (key == '2') toggleMode(GPU_MODE);
    else if (key == '3') toggleMode(CPU_EXT_MODE);
    else if (key == '4') toggleMode(GPU_EXT_MODE);
    else if (key == 'k') show_kernel = !show_kernel;
}

void processSpecialKeys(int key, int x, int y) {
    // other keys (F1, F2, arrows, home, etc.)
    switch (key) {
    case GLUT_KEY_UP:
        my += 1.0f; // Move the camera up
        break;
    case GLUT_KEY_DOWN:
        my -= 1.0f; // Move the camera down
        break;
    case GLUT_KEY_LEFT:
        mx -= 1.0f; // Move the camera left
        break;
    case GLUT_KEY_RIGHT:
        mx += 1.0f; // Move the camera right
        break;
    }
}

void initGL(int argc, char** argv)
{
    // init GLUT and create window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(SCREEN_X, SCREEN_Y);
    glutCreateWindow(TITLE);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glDisable(GL_DEPTH_TEST);

    // View Ortho
    // Sets up the OpenGL window so that (0,0) corresponds to the top left corner,
    // and (SCREEN_X,SCREEN_Y) corresponds to the bottom right hand corner. 
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, SCREEN_X, SCREEN_Y, 0, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.375, 0.375, 0); // Displacement trick for exact pixelization
}


int main(int argc, char** argv) {

    initGL(argc, argv);

    GLint GlewInitResult = glewInit();
    if (GlewInitResult != GLEW_OK) {
        printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
    }

    init();

    glutDisplayFunc(render);
    glutIdleFunc(idle);
    glutMotionFunc(mouseMotion);
    glutMouseFunc(mouse);
    glutKeyboardFunc(processNormalKeys);
    glutSpecialFunc(processSpecialKeys);

    // enter GLUT event processing cycle
    glutMainLoop();

    clean();

    return 1;
}
