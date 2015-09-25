int checkGraph(int * G, int V ) {
	int connected = 1;
	int max = 0;
	int curr;
	for(int i = 0; i < V; i++) {
		if (!connected) {
			printf("Error: Unconnected node %d\n",i-1);
			return -1;
		}
		connected = 0;
		curr = 0;
		for(int j = 0; j < V; j++) {
			if (G[i * V + j] != INF) {
				connected = 1;
				curr++;
			}
			if (G[i * V + j] < 0) {
				printf("Error: Negative edge!!\n");
				return -1;
			}
			if (G[i * V + j] != G[j * V + i]) {
				printf("Error: Graph not symmetric!!\n");
				return -1;
			}
		}
		if (curr > max) {
			max = curr;
		}
	}
	return 0;
}
