//Prints metric closure
void print(int * G, int V) {
	 for(int i = 0; i < V; i++, printf("\n")) {
		  for(int j = 0; j < V; j++) {
		     int out = G[i * V + j];
		     if(out  == INF)
		        printf("%3s " , "INF");
		     else
		        printf("%3d " , out );
		  }
	 }
	 printf("\n");
}

int caclGraphCost(int * G, int V) {
	int sum = 0;
	int curr;
	for(int i = 0; i < V; i++) {
		for(int j = 0; j < V; j++) {
			if (i <= j) {
				continue;
			}
			int curr = G[i * V + j];
			if(curr  == INF) {
			  continue;
			}
			sum += curr;			
		}
	}
	return sum;
}

int isTerminal(int v, int numTer, int * terminals) {
	for(int i = 0; i < numTer; i++)
		if(v == terminals[i])
			return 1;
	return 0;
}

int countNonTerminals(int * G, int V, int numTer, int * terminals) {
	int count = 0;
	for(int i = 0; i < V; i++) {
		if(isTerminal(i,numTer,terminals)) {
			continue;
		}
		int edge = 0;
		for(int j = 0; j < V; j++) {
			if(G[i * V + j] != INF) {
				//printf("i: %d cost: %d \n",i,G[i * V + j]);
				edge = 1;
				break;
			}
		}
		if(edge == 1) {
			count++;
		}
	}
	return count;
}

//Print one star
void printOnestar(int * onestar, int numGroups, int V) {
	for(int i = 0; i < V; i++) {
		 printf("Root%d:  ", i);
		 for(int j = 0; j < numGroups; j++)
			  printf("%3d  ", onestar[i * numGroups + j]);
		 printf("\n");
	}
}

//Print terminals and groups
void printTermGroups(int numTer, int numGroups, int * groups, int * terminals) {
	printf("\nNum of Terminalsu: %d\n",numTer);
	printf("\nTerminal vertices: ");
	for(int i = 0; i < numTer; i++)
		printf("%d ", terminals[i]);
	printf("\n");


	printf("\nNum of Groups: %d\n",numGroups);
	printf("Per group: %d\n",numTer/numGroups);
	for(int i = 0; i < numGroups; i++) {
		printf("Group %d: ", i);
		for(int j = 0; j < numTer/numGroups; j++)
			printf(" %d ", groups[(i * (numTer/numGroups)) + j]);
			printf("\n");
	}
	printf("\n\n");

}

void printPartialStars(int* partialstars, int numGroups, int count) {
	printf("\nParital stars: \n");

	for(int i = 0; i < count; i++) {
		int* curStar = partialstars + (i * (2 + numGroups));
		printf("\tIntermediate: %d # of Groups: %d Group IDs ",curStar[0],curStar[1]);
		for(int j = 0; j < curStar[1]; j++) {
			printf(" %d ", curStar[j+2]);
		}
		printf("\n");
	}
	printf("\n");
}

void copypartialStar(int * A, int  * B, int numGroups, int count) {
	//A - from, B - to
	for(int i = 0; i < count; i++) {
		//get a partial star
		int * curStarA = A + (i * (2 + numGroups));
		int * curStarB = B + (i * (2 + numGroups));

		//copy intermediate and num of groups
		curStarB[0] = curStarA[0];
		curStarB[1] = curStarA[1];

		//copy groups
		for(int j = 0; j < curStarA[1]; j++) {
			curStarB[j+2] = curStarA[j+2];
		}
	}
}

void printSolutionCost(int root, int cost) {
	printf("\nTWOSTAR: \n");
	printf("\tCost: %d Root: %d\n", cost, root);
}

int reconstruct_path(unsigned int n, unsigned int i, unsigned int j, const int * const p, const int * const G, int * patth, int * count)
{
	if (i == j )
		return 0;
	else if ( p[i * n + j] == NONE)
		return INF;
	else
	{
		int path = reconstruct_path(n, i, p[i * n + j], p, G,patth,count);
		if (path == INF) 
			return INF;
		else
			patth[(*count) * 2 + 0] = p[i * n + j];
			patth[(*count) * 2 + 1] = j;
			*count = (*count) + 1;
			return path + G[ p [i * n + j] * n + j];
	}
}
