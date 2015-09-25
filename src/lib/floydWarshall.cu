#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

// CUDA Headers
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Helper definition
#define VAR(v, i) __typeof(i) v=(i)
#define FOR(i, j, k) for (int i = (j); i <= (k); ++i)
#define FORD(i, j, k)for (int i=(j); i >= (k); --i)
#define FORE(i, c) for(VAR(i, (c).begin()); i != (c).end(); ++i)
#define REP(i, n) for(int i = 0;i <(n); ++i)

// CONSTS
#define INF 	1061109567 // 3F 3F 3F 3F
#define CHARINF 63	   // 3F	
#define CHARBIT 8
#define NONE	-1

#define CMCPYHTD cudaMemcpyHostToDevice
#define CMCPYDTH cudaMemcpyDeviceToHost

// CONSTS for compute capability 2.0
#define BLOCK_WIDTH 16
#define WARP 	    32

bool gPrint = false; 	// print graph d or not
bool gDebug = false;	// print more deatails to debug

/** Cuda handle error, if err is not success print error and line in code
*
* @param status CUDA Error types
*/
#define HANDLE_ERROR(err) \
{ \
	if (err != cudaSuccess) \
	{ \
		fprintf(stderr, "%s failed  at line %d \nError message: %s \n", \
			__FILE__, __LINE__ ,cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); \
	} \
}

/**Kernel for wake gpu
*
* @param reps dummy variable only to perform some action
*/
__global__ void wake_gpu_kernel(int reps) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= reps) return;
}

/**Kernel for parallel Floyd Warshall algorithm on gpu
* 
* @param u number vertex of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/
__global__ void fw_kernel(const unsigned int u, const unsigned int n, int * const d, int * const p)
{
	int v1 = blockDim.y * blockIdx.y + threadIdx.y;
	int v2 = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (v1 < n && v2 < n) 
	{
		int newPath = d[v1 * n + u] + d[u * n + v2];
		int oldPath = d[v1 * n + v2];
		if (oldPath > newPath)
		{
			d[v1 * n + v2] = newPath;
			p[v1 * n + v2] = p[u * n + v2];		
		}
	}
}

/** Parallel Floyd Warshall algorithm using gpu
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/
extern "C" void fw_gpu(const unsigned int n, const int * const G, int * const d, int * const p)
{
	int *dev_d = 0;
	int *dev_p = 0;
	cudaError_t cudaStatus;
	cudaStream_t cpyStream;

	// Choose which GPU to run on, change this on a multi-GPU system.
    	cudaStatus = cudaSetDevice(0);
	HANDLE_ERROR(cudaStatus);

	// Initialize the grid and block dimensions here
	dim3 dimGrid((n - 1) / BLOCK_WIDTH + 1, (n - 1) / BLOCK_WIDTH + 1, 1); 
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	if (gDebug) 
	{
		printf("|V| %d\n", n);
		printf("Dim Grid:\nx - %d\ny - %d\nz - %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
		printf("Dim Block::\nx - %d\ny - %d\nz - %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	}

	// Create new stream to copy data	
	cudaStatus = cudaStreamCreate(&cpyStream);
	HANDLE_ERROR(cudaStatus);

	// Allocate GPU buffers for matrix of shortest paths d(G) and predecessors p(G)
	cudaStatus =  cudaMalloc((void**)&dev_d, n * n * sizeof(int));
	HANDLE_ERROR(cudaStatus);
	cudaStatus =  cudaMalloc((void**)&dev_p, n * n * sizeof(int));
	HANDLE_ERROR(cudaStatus);
	
	// Wake up gpu
	wake_gpu_kernel<<<1, dimBlock>>>(32);

        // Copy input from host memory to GPU buffers.
        cudaStatus = cudaMemcpyAsync(dev_d, G, n * n * sizeof(int), CMCPYHTD, cpyStream);
	cudaStatus = cudaMemcpyAsync(dev_p, p, n * n * sizeof(int), CMCPYHTD, cpyStream);

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        cudaStatus = cudaDeviceSynchronize();
        HANDLE_ERROR(cudaStatus);

	cudaFuncSetCacheConfig(fw_kernel, cudaFuncCachePreferL1 );
	FOR(u, 0, n - 1) 
	{
		fw_kernel<<<dimGrid, dimBlock>>>(u, n, dev_d, dev_p);
	}

	// Check for any errors launching the kernel
    	cudaStatus = cudaGetLastError();
	HANDLE_ERROR(cudaStatus);

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	HANDLE_ERROR(cudaStatus);
	
	cudaStatus = cudaMemcpy(d, dev_d, n * n * sizeof(int), CMCPYDTH);
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaMemcpy(p, dev_p, n * n * sizeof(int), CMCPYDTH);
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaFree(dev_d);
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaFree(dev_p);
	HANDLE_ERROR(cudaStatus);

	return;
}

/**
* Print graph G as a matrix
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
*/
void print_graph(const unsigned int n, const int * const G)
{
	FOR(v1, 0, n - 1)
	{
		FOR(v2, 0, n - 1) 
		{	
			if (G[v1 * n + v2] < INF)
				printf("%d ", G[v1 * n + v2]);
			else
				printf("INF ");
		}
		printf("\n");
	}
	printf("\n");
}

/**
* Reconstruct Path
*
* @param i, j id vertex 
* @param G is a the graph G:=(V,E)
* @param p matrix of predecessors p(G)
*/
int reconstruct_path(unsigned int n, unsigned int i, unsigned int j, const int * const p, const int * const G)
{
	if (i == j )
		return 0;
	else if ( p[i * n + j] == NONE)
		return INF;
	else
	{
		int path = reconstruct_path(n, i, p[i * n + j], p, G);
		if (path == INF) 
			return INF;
		else
			return path + G[ p [i * n + j] * n + j];
	}
}

/**
* Check paths
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/
bool check_paths(const unsigned int n, const int * const G, const int * const d, const int * const p)
{
	
	FOR (i, 0, n - 1)
	{
		FOR (j, 0, n - 1)
		{
			int path = reconstruct_path(n, i, j, p, G);
			if (gDebug)
				printf("%d %d %d == %d \n", i, j, path, d[i * n + j]);
			if (path != d[i * n + j])
				return false;
		}
	}

	return true;
}
