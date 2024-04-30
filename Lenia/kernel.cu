
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

#define SCREEN_X 1024
#define SCREEN_Y 768
#define FPS_UPDATE 500
#define TITLE "Lenia"

#define CPU_MODE 1
#define GPU_MODE 2
#define OPENGL_GPU_MODE 3


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
int size = SCREEN_X * SCREEN_Y * sizeof(float4);

#define INF 2e10f

unsigned long long seed = time(NULL); // or any other unique seed value

// Lenia parameters

int R = 1; // because of the CPU capabilities, can not be greater
int T = 10;
float mu = 0.15f;
float omega = 0.016f;
int alpha = 4;
#define B 1 // rank for the pics

float4* lenia_pixels;


float4* randomPixels() {
	float4* p = (float4*)malloc(SCREEN_X * SCREEN_Y * sizeof(float4));
	for (int i = 0; i < SCREEN_X * SCREEN_Y; i++) {
		if (i % (10 * SCREEN_X) == 0) srand(i);
		float random_value = (float)rand() / RAND_MAX; // Generate a random float between 0 and 1
		p[i].x = 0.0f;
		p[i].y = random_value; 
		p[i].z = 0.0f;
		p[i].w = 1.0f;
	}
	return p;
}

float4* zeroPixels() {
	float4* p = (float4*)malloc(size);
	for (int i = 0; i < SCREEN_X * SCREEN_Y;i++) {
		p[i].x = 0.0f;
		p[i].y = 0.0f;
		p[i].z = 0.0f;
		p[i].w = 1.0f;
	}
	return p;
}

void initCPU()
{
	time_t t;

	// Intializes random number generator
	srand((unsigned)time(&t));

	lenia_pixels = randomPixels();
	pixels2 = zeroPixels();

}

void cleanCPU()
{
	free(lenia_pixels);
	free(pixels2);
}

// ------------------------- LENIA -------------------------------------------------------------------------------------

/*
	exponential core function
	pre-condition : r between 0 and 1
	post-condition : return value between O and 1
*/
float kernel_core_exp(float r)
{
	return exp(alpha - (alpha / (4 * r * (1 - r))));
}

/*
	kernel shell
	pre-conditions : r between 0 and 1 ; beta in [0;1] dim B
	post-condition : return value between 0 and 1
*/
/*float kernel_shell(float r, float beta[B])
{
	int index = static_cast<int>(B * r);
	double fraction = B * r - index;

	double peak_height = beta[index];

	return peak_height * kernel_core_exp(fraction);
}*/

/*
	Normalization of the kernel
	pre-condition : n in the neighbourhood, at the indexes i and j
	post-condition : 
*/
float normalized_kernel(int x, int y, float beta[B])
{
	float norm_n = sqrt(x*x + y*y);
	float Ks_val = kernel_core_exp(norm_n); //kernel_shell(norm_n, beta)

	float sum = 0.0f;
	// sum of the Ks of the neighbourhood (ie where norm(x) <= R)
	for (int i = x-R; i <= x+R;i++) {
		for (int j = y-R; j <= y+R;j++) {
			if (i != x || j != y) {
				int wrappedI = (i + SCREEN_Y) % SCREEN_Y;
				int wrappedJ = (j + SCREEN_X) % SCREEN_X;
				sum += kernel_core_exp(sqrt(wrappedI * wrappedI + wrappedJ * wrappedJ));//kernel_shell(sqrt(wrappedI * wrappedI + wrappedJ * wrappedJ), beta);
			}
		}
	}

	return Ks_val / sum;
}

/*
	Potential distribution U_t(x) - Local rule
	Pre-condition : in the grid at indexes x and y
	Post-condition : return value between 0 and 1
*/
float potential_distribution(int x, int y)
{
	float sum = 0.0f;
	float beta[B];
	beta[0] = 1.0f;

	for (int i = x - R; i <= x + R;i++) {
		for (int j = y - R; j <= y + R;j++) {
			if (i != x || j != y) {
				// calculate the wrapped index
				int wrappedI = (i + SCREEN_Y) % SCREEN_Y;
				int wrappedJ = (j + SCREEN_X) % SCREEN_X;

				sum += normalized_kernel(x, y, beta) * lenia_pixels[wrappedI * SCREEN_X + wrappedJ].y;
			}
		}
	}

	return sum;
}


float growth_function_exp(float u)
{
	float dividende = (2.0f * omega * omega);
	return 2.0f * exp(-((u - mu) * (u - mu)) / dividende) - 1.0f;
}

void lenia_basic_CPU()
{
	// for each pixel
	for (int i = 0; i < SCREEN_Y; i++) {
		for (int j = 0; j < SCREEN_X; j++) {
			float c_t = lenia_pixels[i * SCREEN_X + j].y; // field C at time step t
			//printf("c_t = %f - ", c_t);
			float c_tdt; // field C at time step t + delta(t) ; (delta (t) = 1/T)

			// 1st step : convolution operation, multiplication with the kernel
			float u_t = potential_distribution(i, j);
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
			pixels2[i * SCREEN_X + j].y = c_tdt;
		}
	}
}


void calculate() {
	frame++;
	int timecur = glutGet(GLUT_ELAPSED_TIME);

	if (timecur - timebase > FPS_UPDATE) {
		char t[200];
		char* m = "";
		switch (mode)
		{
		case CPU_MODE: m = "CPU mode"; break;
		//case GPU_MODE: m = "GPU mode - normal"; break;
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
	//case GPU_MODE: bugsCPU(); break;
	//case OPENGL_GPU_MODE: bugsCPU(); break;
	}
}

void idle()
{
	glutPostRedisplay();
}


void render()
{
	calculate();

	glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, lenia_pixels);

	glutSwapBuffers();
}

void clean()
{
	switch (mode)
	{
	case CPU_MODE: cleanCPU(); break;
	case GPU_MODE: cleanCPU(); break;
	case OPENGL_GPU_MODE: cleanCPU(); break;
	}

}

void init()
{
	switch (mode)
	{
	case CPU_MODE: initCPU(); break;
	case GPU_MODE: initCPU(); break;
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
	//else if (key == '2') toggleMode(GPU_MODE);
	//else if (key == '3') toggleMode(OPENGL_GPU_MODE);
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
