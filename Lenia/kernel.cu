
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

#define SCREEN_X 256
#define SCREEN_Y 256
#define FPS_UPDATE 500
#define TITLE "Lenia"

#define CPU_MODE 1
#define GPU_MODE 2
#define OPENGL_GPU_MODE 3

int width_grid = 128;
int height_grid = 128;

int block_dim_x = 16;
int block_dim_y = 16;
int grid_dim_x = (width_grid + block_dim_x - 1) / block_dim_x;
int grid_dim_y = (height_grid + block_dim_y - 1) / block_dim_y;

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
int mode = CPU_MODE;
int frame = 0;
int timebase = 0;

float4* pixels2;
int size = width_grid * height_grid * sizeof(float4);

#define INF 2e10f

unsigned long long seed = time(NULL); // or any other unique seed value

// Lenia parameters

#define R 13
#define T 10
#define mu 0.15f
#define sigma 0.017f
#define alpha 4
#define B 1 // rank for the pics

float4* lenia_pixels;
float* n_kernel;
float k_s_norm;
float beta[B];

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

float4* randomPixels() {
	float4* p = (float4*)malloc(size);
	for (int i = 0; i < width_grid * height_grid; i++) {
		if (i % (10 * width_grid) == 0) srand(i);
		float random_value = (float)rand() / RAND_MAX; // Generate a random float between 0 and 1
		float test_alive = (float)rand() / RAND_MAX;

		if (test_alive < 0.77f) {
			random_value = 0.0f;
		}

		p[i].x = 0.0f;
		p[i].y = random_value;
		p[i].z = 0.0f;
		p[i].w = 1.0f;
	}
	return p;
}

float4* randomPixelsCenter() {
	float4* p = (float4*)malloc(size);
	for (int i = 3 * height_grid / 8; i < 5 * height_grid / 8; i++) {
		for (int j = 3 * width_grid / 8; j < 5 * width_grid / 8; j++) {
			int index = i * width_grid + j;

			if (index % (10 * width_grid) == 0) srand(index);
			float random_value = (float)rand() / RAND_MAX; // Generate a random float between 0 and 1
			p[index].x = 0.0f;
			p[index].y = random_value;
			p[index].z = 0.0f;
			p[index].w = 1.0f;
		}

	}
	return p;
}

float4* orbium() // pas complet du tout
{
	float4* p = zeroPixels();
	p[6].y = 0.1f;
	p[7].y = 0.14f;
	p[8].y = 0.1f;
	p[11].y = 0.03f;
	p[12].y = 0.03f;
	p[15].y = 0.3f;
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


void cleanCPU()
{
	free(lenia_pixels);
	free(pixels2);
	free(n_kernel);
}


void initGPU()
{
	time_t t;

	// Intializes random number generator
	srand((unsigned)time(&t));

	lenia_pixels = randomPixels();
	pixels2 = zeroPixels();
	checkCudaErrors(cudaMalloc((void**)&d_grid1, size));
	checkCudaErrors(cudaMalloc((void**)&d_grid2, size));

}

void cleanGPU()
{
	free(lenia_pixels);
	free(pixels2);
	cudaFree(d_grid1);
	cudaFree(d_grid2);
}

// ------------------------- LENIA -------------------------------------------------------------------------------------

/*
	exponential core function
	pre-condition : r between 0 and 1
	post-condition : return value between 0 and 1
*/
__host__ __device__ float kernel_core_exp(float r)
{
	float k = (4.0f * r * (1.0f - r));
	float k_core = exp(alpha * (1.0f - 1.0f/k));
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

int emod(int a, int b)
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
float kernel_shell(float n) //OK
{
	int index = emod(floor((float)B * n), B);

	double peak_height = beta[index];

	float u = fmod(float(B * n), 1.0f);

	return peak_height * kernel_core_exp(u);
}

double toric_distance(int x1, int y1, int x2, int y2, int width, int height) {
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
__host__ float normalized_kernel(int x, int y, int width_grid, int height_grid)
{
	int center_x = width_grid / 2;
	int center_y = height_grid / 2;

	float norm_n = toric_distance(center_x, center_y, x, y, width_grid, height_grid) / R;
	float Ks_val = 0.0f;
	if (toric_distance(center_x, center_y, x, y, width_grid, height_grid) <= R) {
		Ks_val = kernel_shell(norm_n);
	}

	return Ks_val;
}


void init_kernel() {

	n_kernel = (float*)malloc(width_grid * height_grid * sizeof(float));

	for (int x = 0; x < height_grid; x++) {
		for (int y = 0; y < width_grid; y++) {
			float val = normalized_kernel(x, y, width_grid, height_grid);
			n_kernel[x * width_grid + y] = val;
		}
	}
}

float init_k_s_norm() {
	float acc = 0;
	for (int i = -R; i <= R; i++) {
		for (int j = -R; j <= R; j++) {
			float r = sqrt(i * i + j * j) / R;
			if (r <= 1.0f) {  // within the radius
				acc += kernel_shell(r);
			}
		}
	}
	return acc;
}


void initCPU()
{
	time_t t;
	beta[0] = 1.0f;
	//beta[1] = 1.0f/3.0f;
	//beta[2] = 7.0f/12.0f;

	// Intializes random number generator
	srand((unsigned)time(&t));

	lenia_pixels = randomPixelsCenter();
	pixels2 = zeroPixels();

	init_kernel();

	k_s_norm = init_k_s_norm();

	printf("ksnorm = %f\n", k_s_norm);
}


/*
	Potential distribution U_t(x) - Local rule
	Pre-condition : in the grid at indexes x and y
	Post-condition : return value between 0 and 1
*/
__host__ float potential_distribution(int x, int y, int width, int height) {
	float sum = 0.0f;

	for (int i = -R; i <= R; i++)
	{
		for (int j = -R; j <= R; j++)
		{

			int ni = (x + i + height) % height;
			int nj = (y + j + width) % width;
			float r = sqrt(i * i + j * j) / R;
			if (r <= 1.0f) {  // within the radius
				sum += kernel_shell(r) * lenia_pixels[ni * width + nj].y;
			}

		}
	}

	return sum / k_s_norm;
}



void lenia_basic_CPU()
{
	// for each pixel
	for (int i = 0; i < height_grid; i++) {
		for (int j = 0; j < width_grid; j++) {
			float c_t = lenia_pixels[i * width_grid + j].y; // field C at time step t
			//printf("c_t = %f - ", c_t);
			float c_tdt; // field C at time step t + delta(t) ; (delta (t) = 1/T)

			// 1st step : convolution operation, multiplication with the kernel
			float u_t = potential_distribution(i, j, width_grid, height_grid);
			//printf("u_t = %f - ", u_t);
			// 2nd step : growth mapping
			float g_t = growth_function_exp(u_t);
			//printf("g_t = %f - ", g_t);
			// 3rd step : add the growth to the existing value
			float dt = 1.0f / float(T);
			c_tdt = c_t + dt * g_t;
			//printf("c_t step 3 = %f - ", c_tdt);
			// 4th step : clip the result to be in range from 0 to 1
			if (c_tdt < 0.0f) c_tdt = 0.0f;
			if (c_tdt > 1.0f) c_tdt = 1.0f;

			//printf("c_tdt = %f\n", c_tdt);

			// assign the value to a temporary table
			pixels2[i * width_grid + j].y = c_tdt;
		}
	}
}


/*__global__ void lenia(float4* d_grid_old, float4* d_grid_new, int width, int height)
{
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	if (indexX < width && indexY < height) {
		int index = indexY * width + indexX;

		float c_t = d_grid_old[index].y; // field C at time step t
		float c_tdt; // field C at time step t + delta(t) ; (delta (t) = 1/T)

		// 1st step : convolution operation, multiplication with the kernel

		float sum = 0.0f;
		float beta[B];
		beta[0] = 1.0f;

		float n_kernel = normalized_kernel(indexX, indexY, beta, width, height);

		for (int i = indexX - R; i <= indexX + R;i++) {
			for (int j = indexY - R; j <= indexY + R;j++) {
				if (i != indexX || j != indexY) {
					// calculate the wrapped index
					int wrappedI = (i + height) % height;
					int wrappedJ = (j + width) % width;

					sum += n_kernel * d_grid_old[wrappedI * width + wrappedJ].y;
				}
			}
		}
		float u_t = sum;

		// 2nd step : growth mapping
		float g_t = growth_function_exp(u_t);

		// 3rd step : add the growth to the existing value
		float dt = 1.0f / float(T);
		c_tdt = c_t + dt * g_t;

		// 4th step : clip the result to be in range from 0 to 1
		if (c_tdt < 0.0f) c_tdt = 0.0f;
		if (c_tdt > 1.0f) c_tdt = 1.0f;

		// assign the value
		d_grid_new[index].y = c_tdt;
	}
}


void lenia_basic_GPU()
{
	dim3 dimBlock(block_dim_x, block_dim_y);
	dim3 dimGrid(grid_dim_x, grid_dim_y);

	cudaError_t err;

	if (tab_1_used)
	{
		// send lenia pixels to device
		checkCudaErrors(cudaMemcpy(d_grid1, lenia_pixels, size, cudaMemcpyHostToDevice));

		// do treatments
		lenia << <dimGrid, dimBlock >> > (d_grid1, d_grid2, width_grid, height_grid);
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error: % s\n", cudaGetErrorString(err));
		}

		// fetch grid 2
		checkCudaErrors(cudaMemcpy(pixels2, d_grid2, size, cudaMemcpyDeviceToHost));
	}
	else {
		// send lenia pixels to device
		checkCudaErrors(cudaMemcpy(d_grid2, pixels2, size, cudaMemcpyHostToDevice));

		// do treatments
		lenia << <dimGrid, dimBlock >> > (d_grid2, d_grid1, width_grid, height_grid);
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error: % s\n", cudaGetErrorString(err));
		}

		// fetch grid 1
		checkCudaErrors(cudaMemcpy(lenia_pixels, d_grid1, size, cudaMemcpyDeviceToHost));
	}

}*/


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
			//case OPENGL_GPU_MODE: m = "GPU mode - OpenGL interoperability"; break;
		}
		sprintf(t, "%s:  %s, %.2f FPS", TITLE, m, frame * 1000 / (float)(timecur - timebase));
		glutSetWindowTitle(t);
		timebase = timecur;
		frame = 0;
	}

	switch (mode)
	{
	case CPU_MODE:
		lenia_basic_CPU();
		lenia_pixels = pixels2;
		break;
	case GPU_MODE:
		//lenia_basic_GPU();
		break;
		//case OPENGL_GPU_MODE: bugsCPU(); break;
	}
}

void idle()
{
	glutPostRedisplay();
}

void draw_pixels_zoomed()
{
	// Calculate the size of each grid cell on the screen
	int cell_size_x = SCREEN_X / width_grid;
	int cell_size_y = SCREEN_Y / height_grid;

	for (int i = 0; i < height_grid; i++)
	{
		for (int j = 0; j < width_grid; j++)
		{
			float4 color;
			// Get the color of the current pixel in the grid
			if (mode == GPU_MODE && !tab_1_used) {
				color = pixels2[i * width_grid + j];
			}
			else {
				color = lenia_pixels[i * width_grid + j];
			}

			// Draw a rectangle representing the grid cell
			glBegin(GL_QUADS);
			glColor4f(color.y, color.y, color.y, color.w);
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
			float color_val = n_kernel[i * width_grid + j];
			//printf("%f;", n_kernel[i * width_grid + j]);
			// Draw a rectangle representing the grid cell
			glBegin(GL_QUADS);
			glColor4f(color_val, color_val, color_val, 1.0f);
			glVertex2i(j * cell_size_x, i * cell_size_y); // Top-left corner
			glVertex2i((j + 1) * cell_size_x, i * cell_size_y); // Top-right corner
			glVertex2i((j + 1) * cell_size_x, (i + 1) * cell_size_y); // Bottom-right corner
			glVertex2i(j * cell_size_x, (i + 1) * cell_size_y); // Bottom-left corner
			glEnd();
		}
	}
	//printf("\n");
}


void render()
{
	calculate();

	if (show_kernel) {
		draw_kernel();
	}
	else {
		if (SCREEN_X == width_grid && SCREEN_Y == height_grid) {
			glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, lenia_pixels);
		}
		else {
			draw_pixels_zoomed();
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
	case OPENGL_GPU_MODE: cleanCPU(); break;
	}

}

void init()
{
	tab_1_used = true;

	switch (mode)
	{
	case CPU_MODE: initCPU(); break;
	case GPU_MODE: initGPU(); break;
	case OPENGL_GPU_MODE: initCPU(); break;
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
	//else if (key == '3') toggleMode(OPENGL_GPU_MODE);
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

	init();

	glutDisplayFunc(render);
	glutIdleFunc(idle);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);

	GLint GlewInitResult = glewInit();
	if (GlewInitResult != GLEW_OK) {
		printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
	}

	// enter GLUT event processing cycle
	glutMainLoop();

	clean();

	return 1;
}
