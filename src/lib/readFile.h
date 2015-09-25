//Reads graph, terminal vertices and groups from file, also allocates memory for the parent process
void readFile(int ** D, int ** G, int ** P, int ** term, int ** groups, int * V, int * E, int * numTer, int * numGroups) {
	unsigned int v1, v2, w;
	
	//printf("\nLoading graph from file...\n");

	//get size of graph	
	scanf("%d %d", V, E);

	if(debug) {
		printf("V: %d  E: %d\n",*V,*E);
	}

	//allocate memory
	*D = (int *) malloc(sizeof(int) * (*V) * (*V));
	*G = (int *) malloc(sizeof(int) * (*V) * (*V));
	*P = (int *) malloc(sizeof(int) * (*V) * (*V));

	// Init Data for the graph G and p
	//memset(P, NONE, sizeof(int) * (*V) * (*V));

	//initialize graph to INF
	for(int i = 0; i < (*V); i++) {
		for(int j = 0; j < (*V); j++) {
			(*G)[i * (*V) + j] = INF;
			(*P)[i * (*V) + j] = NONE;
		}
	}
	
	//read graph and number of terminals
	for(int e = 0; e < (*E); e++) {
		scanf("%d %d %d", &v1, &v2, &w);
		(*G)[(v1 - 1) * (*V) + (v2 - 1)] = w;
		(*G)[(v2 - 1) * (*V) + (v1 - 1)] = w;
		if ((v1 - 1) != (v2 - 1)) {
			(*P)[(v1 - 1) * (*V) + (v2 - 1)] = (v1 - 1);
			(*P)[(v2 - 1) * (*V) + (v1 - 1)] = (v2 - 1);
		}
	}

	//read terminals
	scanf("%d",numTer);
	*term = (int *) malloc(sizeof(int) * (*numTer));
	*groups = (int *) malloc (sizeof(int) * (*numTer));

	for(int i = 0; i < (*numTer); i++) {
		int v;
		scanf("%d", &v);
		(*term)[i] = v - 1;
	}

	//read groups
	scanf("%d", numGroups);
	for(int i = 0; i < (*numGroups); i++) {
		for(int j = 0; j < (*numTer)/(*numGroups); j++) {
		  int v;
		  scanf("%d", &v);
		  (*groups)[(i * ((*numTer)/(*numGroups))) + j] = v - 1;
		}
	}
}
