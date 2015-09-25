//Serial floyd warshall algorithm for APSP problem
void floydWarshall(int V, int *graph, int *dist ) {
 	int i, j, k;
	for (i = 0; i < V; i++)
  		for (j = 0; j < V; j++)
     		dist[i * V + j] = graph[i * V + j]; //copy graph to solution matrix
 
	for (k = 0; k < V; k++)
   	for (i = 0; i < V; i++) //pick source
      	for (j = 0; j < V; j++) //pick destinations one by one
         	if (dist[i * V + k] + dist[k * V + j] < dist[i * V + j])
         		dist[i * V + j] = dist[i * V + k] + dist[k * V + j];
}



/**
 * Constructs matrix P0
 * @param d matrix of lengths
 * @return P0
 */
void constructInitialMatixOfPredecessors(int * D, int * P, int V) {
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (D[i * V + j] != 0 && D[i * V + j] != INF) {
                P[i * V + j] = i;
            } else {
                P[i * V + j] = -1;
            }
        }
    }
}


/**
 * Floyd-Warshall algorithm. Finds all shortest paths among all pairs of nodes
 * @param d matrix of distances (INF represents positive infinity)
 * @return matrix of predecessors
 */
void floydWarshallWithPath(int V, int * G, int * D, int * P) {
	int i, j, k;
	for (i = 0; i < V; i++)
  		for (j = 0; j < V; j++)
     		D[i * V + j] = G[i * V + j]; //copy graph to solution matrix

    //constructInitialMatixOfPredecessors(D,P,V);
    for (k = 0; k < V; k++) {
        for (i = 0; i < V; i++) {
            for (j = 0; j < V; j++) {
                /*if (D[i * V + k] == INF || D[k * V + j] == INF) {
                    continue;                  
                }*/
                
                if (D[i * V + j] > D[i * V + k] + D[k * V + j]) {
                    D[i * V + j] = D[i * V + k] + D[k * V + j];
                    P[i * V + j] = P[k * V + j];
                }

            }
        }
    }
}
