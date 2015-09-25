//Library headers
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <getopt.h>
#include <string.h>
#include <omp.h>
#include <time.h>
//Library for sched_getcpu()
#define _GNU_SOURCE 
#include <utmpx.h>

//Constant
#define INF 1061109567 	//3F3F3F3F
#define NONE	-1			//for predecessors	
#define CHARINF 63	   //3F

//Global variables
bool gprint = false; // print graph and metric closure -o
bool debug = false;	// print more deatails for debugging -d
bool parallel = false; //construct metric closure in serial or parallel -p
bool build = false; //build solution -b
bool stpFile = false;
//File headers
#include "lib/utils.h"
#include "lib/readFile.h"
#include "lib/readFile2.h"
#include "lib/floydSerial.h"
#include "lib/onestar.h"
#include "lib/twostar.h"
#include "lib/buildsolution.h"
 
//Function declaration
int sched_getcpu(void);
void fw_gpu(const unsigned int n, const int * const G, int * const d, int * const p);

//Main
int main(int argc, char *argv[])
{	
	//initialize MPI variables
	int numProc, procId;
	MPI_Status status;
	MPI_Request request;

	//initialize MPI environment
	MPI_Init(&argc,&argv);  
	MPI_Comm_size(MPI_COMM_WORLD,&numProc);  
	MPI_Comm_rank(MPI_COMM_WORLD,&procId);

	//graph variables
	unsigned int V, E, numTer, numGroups;
	int *D, *G, *P, *C, *term, *groups, *D_sub, *onestar, *onestar_sub, *onestar_V, *onestar_sub_V;

	//solution variables
	int MINIMUM, overall_min;
	struct Solution solution;
	struct Solution minSolution;
	struct TwoStar twostar;
	//struct Solution solutionTree

	//variables for mapping roots to processes
	int *pars;
	int perParent;
	int perChild;
	
	clock_t t_gpu, t_cpu;
	//variables for calculating time
	double starttime, endtime, time_taken;	
	
	//MPI_WTIME_IS_GLOBAL = 1;
	//MPI_Barrier(MPI_COMM_WORLD);
	//t_tot = MPI_Wtime();

	//mapping of processes to CPU cores
	if(debug) {
		printf("ProcID = %d CpuID = %d\n", procId, sched_getcpu());
	}

	/*--------------------------------------------Parent process------------------------------------------------*/
	if(!procId)  {
		int r;
		while ((r = getopt(argc, argv, "odpbt")) != -1) { //command line args
			switch(r)
			{
				case 'o':
					gprint = true;
					break;
				case 'd':
					debug = true;
					break;
				case 'p':
					parallel = true;
					break;
				case 'b':
					build = true;
					break;
				case 't':
					stpFile = true;
					break;
				default:
					//printUsage();
					exit(1);
			}
		}

		//read graph from file and allocate memory
		if(stpFile) {
			readFile2(&D, &G, &P, &C, &term, &groups, &V, &E, &numTer, &numGroups);
		}
		else {
			readFile(&D, &G, &P, &term, &groups, &V, &E, &numTer, &numGroups);
		}
		//broadcast size variables
		MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&numTer, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&numGroups, 1, MPI_INT, 0, MPI_COMM_WORLD);

		//buffer for combined onestar cost matrix
		onestar = (int *) malloc(sizeof(int) * V * numGroups);
		onestar_V = (int *) malloc(sizeof(int) * V * numGroups);

		//broadbast groups
		MPI_Bcast(groups, numTer, MPI_INT, 0, MPI_COMM_WORLD);

		//validate number of processes
		if(!validNumProc(V, numProc)) {
			printf("Error: More number of compute nodes than needed.\n");
			MPI_Finalize();
			return 0;
		}

		//calculate roots per process
		pars = calcLaunchPar(numProc, V);
		perParent = pars[1];
		perChild = pars[0];

		//output for debug
		if(debug) {	
			printf("Number of vertices: %d\n", V);
			printf("Parent process gets: %d\n", pars[1]);
			printf("%d child processes get: %d\n", numProc - 1, pars[0]);
		}

		//construct metric closure
		if(parallel) {
			if(debug) {
				printf("Construction metric closure on the GPU...\n");			
			}
			//MPI_Barrier(MPI_COMM_WORLD);
			t_gpu = clock();
			fw_gpu(V, G, D, P);
			//MPI_Barrier(MPI_COMM_WORLD);
			t_gpu = clock() - t_gpu;
			t_gpu = (double)t_gpu/CLOCKS_PER_SEC;
 			
		} else{
			if(debug) {
				printf("Construction metric closure on the CPU...\n");			
			}
			//floydWarshall(V, G, D);
			floydWarshallWithPath(V,G,D,P);
		}

		//broadcast metric closure - non-blocking
		//MPI_Ibcast(D,V*V, MPI_INT, 0, MPI_COMM_WORLD,&request);
		MPI_Bcast(D,V*V, MPI_INT, 0, MPI_COMM_WORLD);

		//output for debug
		if(gprint) {
			printf("Graph:\n");
			print(G,V);
			printf("Metric Closure:\n");
			print(D,V);
			printf("Predecessors:\n");
			print(P,V);
			printTermGroups(numTer,numGroups,groups,term);
		}

		//reciveing buffer for distributing some rows of metric closure
		D_sub = (int *) malloc(sizeof(int) * V * perChild);

		//buffer for sub onestar matrix in each process
		onestar_sub = (int *) malloc(sizeof(int) * perChild * numGroups);
		onestar_sub_V= (int *) malloc(sizeof(int) * perChild * numGroups);
		
		MPI_Barrier(MPI_COMM_WORLD);
		t_cpu = clock();
		//construct one star
		onestarWrapper(V,numTer,perChild,perParent,numProc,procId,numGroups,D,D_sub,onestar,onestar_sub,onestar_V, onestar_sub_V,groups);
	
		//broadbast onestar
		MPI_Bcast(onestar, V * numGroups, MPI_INT, 0, MPI_COMM_WORLD);

		//output onestar
		if(gprint) {
			printf("One star cost: \n");
			printOnestar(onestar,numGroups,V);
			printf("One star terminals: \n");
			printOnestar(onestar_V,numGroups,V);
		}

		//check if metric closure broadcast is done
		//MPI_Wait(&request, &status);
		//construct two star
		twostarwrapper(V,numGroups,perChild,perParent,numProc,procId,D,onestar,&solution,&twostar);
		
		//get minimum from all using reduction
		MPI_Reduce(&solution,&minSolution,1,MPI_2INT,MPI_MINLOC,0,MPI_COMM_WORLD);

		//ouput overall minimum cost
		//if(!build) {		
			//printf("\nOVERALL MINIMUM STEINER COST: %d Root: %d\n\n", minSolution.cost, minSolution.root);
		//}
		
		t_cpu = clock() - t_cpu;
		t_cpu = (double)t_cpu/CLOCKS_PER_SEC;
		//if(build) {
		buildWrapper(minSolution,V,numGroups,P,G,D,onestar,onestar_V,term,numTer,perParent,perChild,numProc,procId,&twostar);
		//}
		
		//t_cpu = MPI_Wtime() - t_cpu;
	}//end parent process
	

	/*--------------------------------------------Child processes------------------------------------------------*/
	if(procId) {
		//broadcast size variables
		MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&numTer, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&numGroups, 1, MPI_INT, 0, MPI_COMM_WORLD);

		//allocate memory
		D = (int *) malloc(sizeof(int) * V * V);
		groups = (int *) malloc (sizeof(int) * numTer);

		//buffer for combined onestar cost matrix
		onestar = (int *) malloc(sizeof(int) * V * numGroups);
		onestar_V = (int *) malloc(sizeof(int) * V * numGroups);

		//broadbast groups
		MPI_Bcast(groups, numTer, MPI_INT, 0, MPI_COMM_WORLD);


		//validate number of processes
		if (!validNumProc(V, numProc)) {
			MPI_Finalize();
			return 0;
		}

		//calculate number of roots per process
		pars = calcLaunchPar(numProc, V);
		perParent = pars[1];
		perChild = pars[0];

		//broadcast metric closure
		//MPI_Ibcast(D,V*V, MPI_INT, 0, MPI_COMM_WORLD,&request);
		MPI_Bcast(D,V*V, MPI_INT, 0, MPI_COMM_WORLD);


		//buffer for reciving rows of metric closure
		D_sub = (int *) malloc(sizeof(int) * V * perChild);

		//buffer for sub onestar matrix in each process
		onestar_sub = (int *) malloc(sizeof(int) * perChild * numGroups);
		onestar_sub_V = (int *) malloc(sizeof(int) * perChild * numGroups);
		
		MPI_Barrier(MPI_COMM_WORLD);
		//construct one star for assigned roots
		onestarWrapper(V,numTer,perChild,perParent,numProc,procId,numGroups,D,D_sub,onestar,onestar_sub,onestar_V, onestar_sub_V,groups);

		//recieve onestar
		MPI_Bcast(onestar, V * numGroups, MPI_INT, 0, MPI_COMM_WORLD);

		//check if metric closure broadcast is done
		//MPI_Wait(&request, &status);

		//construct two star
		twostarwrapper(V,numGroups,perChild,perParent,numProc,procId,D,onestar,&solution,&twostar);

		//get minimum of all
		MPI_Reduce(&solution,&minSolution,1,MPI_2INT,MPI_MINLOC,0,MPI_COMM_WORLD);		

		buildWrapper(minSolution,V,numGroups,P,G,D,onestar,onestar_V,term,numTer,perParent,perChild,numProc,procId,&twostar);
	
		
	}//end child processes

	if(!procId){
		printf("GPU took %f seconds\n", t_gpu);
		printf("CPU took %f seconds\n", t_cpu);
	}

	MPI_Finalize();
	return 0;
}
