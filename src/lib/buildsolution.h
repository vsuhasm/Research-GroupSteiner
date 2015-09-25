#include <omp.h>

void drawEdge(int i, int j, int V, int * S, int * G) {
	S[i * V + j] = G[i * V + j];
	S[j * V + i] = G[j * V + i];
}

//gets path from i to j using predecessors matrix, stores in in 'path'
int getPath(int i, int j, int V, int * path, int * P, int * G) {
	int count = 0;
	int final = reconstruct_path(V, i,j, P, G,path,&count);
	return count;
}


void buildWrapper(struct Solution minSolution, int V, int numGroups, int * P,int * G, int * D, int * onestar, int * onestar_V, int * terminals, int numTer, int perParent, int perChild, int numProc, int procId, struct TwoStar * twostar) {
	
	int root, cost, count, *partialStar1;
	int minProc;
	int * path, * S;
	root = minSolution.root;
	cost = minSolution.cost;

	if(!procId) {
		minProc = getProcId(minSolution.root,perChild, perParent,numProc,V);
		//printf("Min Proc: %d \n", minProc);
		
		MPI_Bcast(&minProc,1,MPI_INT,0,MPI_COMM_WORLD);
		if(minProc == 0) {
			count = twostar->numPar;
			partialStar1 = twostar->partialstars;
		}
		else {
			MPI_Recv(&(twostar->numPar), 1, MPI_INT, minProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(twostar->partialstars,(2 + numGroups) * V, MPI_INT, minProc, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			
			count = twostar->numPar;		
			partialStar1 = twostar->partialstars;
		} 			
		
		// variables
		S = (int *) malloc(sizeof(int) * V * V);
		
		#pragma omp parallel for
		//initialize graph to INF
		for(int i = 0; i < V; i++) {
			for(int j = 0; j < V; j++) {
				S[i * V + j] = INF;
			}
		}
	
		printSolutionCost(root, cost);
		printPartialStars(partialStar1,numGroups,count);
	
		path = (int *) malloc(sizeof(int) * 2 * V);
		for(int i = 0; i < count; i++) { //for each partial star
			int * curStar = partialStar1 + (i * (2 + numGroups));
			int interm = curStar[0];
			int c = getPath(root,interm,V,path,P,G);
			//printf("INTERMEDIATE: %d \n", interm);
			//printf("\troot to inter count: %d\n", c);
			for(int j = 0; j < c; j++) {
				int s = path[j * 2 + 0];
				int d = path[j * 2 + 1];
				//printf("\t\t\tS: %d  D: %d\n",s,d);
				drawEdge(s, d , V, S, G);
			}
			//printf("\n");
			for(int j = 0; j < curStar[1]; j++) { //for each groupID
				int term = onestar_V[interm * numGroups + curStar[2+j]];
				int cc = getPath(interm, term,V,path,P,G);
				//printf("\tGROUP: %d TERM: %d \n", curStar[2+j], term);
				//printf("\t\tinterm to term count: %d\n", cc);
				for(int k = 0; k < cc; k++) {
					int s = path[k * 2 + 0];
					int d = path[k * 2 + 1];
					//printf("\t\t\tS: %d  D: %d\n",s,d);
					drawEdge(s, d , V, S, G);
				}
			}
			//printf("\n\n");
		}
		//print(S,V);
		printf("Final Graph Cost: %d\n", caclGraphCost(S,V));
		//printf("NonTer part of solution: %d\n",countNonTerminals(S,V,numTer, terminals));
	}//parent process


	else {
		MPI_Bcast(&minProc,1,MPI_INT,0,MPI_COMM_WORLD);
		if(minProc == 0) {
			;
		}	
		else if (procId == minProc) {
			MPI_Send(&(twostar->numPar),1,MPI_INT,0,0,MPI_COMM_WORLD);
			MPI_Send(twostar->partialstars,(2 + numGroups) * V,MPI_INT,0,0,MPI_COMM_WORLD);			
		}
	}
}
