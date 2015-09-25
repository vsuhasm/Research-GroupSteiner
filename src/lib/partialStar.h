#define MAX_NORM 147483640.0
#include <omp.h>

struct InterInfo
{	
	int * groupss; //groups spanned
	int interm; //intermediate vetex ID
   float norm; //norm of partial star
   int numGroups; //num of groups spanned
};

void sort(int, int, int*, int*, int, int);
float norm(int, int, int, int*, int, int*, int*, int, int);
void partialStarLoop(int, int*, int*, int, int, int, int*,struct InterInfo *);
int isAvailable(int, int *);
void bubble_sort2(int * groupIds, int n);

void partialStar(int src, int* groupIds, int* vertToGroup, int numGroups, int remGroups, int V, int* rTOv,int * groupsSpanned, int* intermSet ) {
	//rTOv is r^th row of the metric closure
	struct InterInfo * normV = (struct InterInfo *)  malloc(sizeof(struct InterInfo) * V); //array of structure

	for(int i = 0; i < V; i++) { //allocate memory
		normV[i].groupss =  (int *) malloc(sizeof(int) * remGroups);
	}

	partialStarLoop(src,groupIds,vertToGroup,numGroups,remGroups,V,rTOv,normV); //return all v with thier norms and groups spanned

	int firstTime = 1;
	int minV;
	float minNorm;
	for(int v = 0; v < V; v++) { //find v with minimum norm
		if(v == src)
			continue;
		if(!isAvailable(v,intermSet))
			continue;
		if(firstTime) {
			minV = v;
			minNorm = normV[minV].norm;
			firstTime = 0;
			continue;
		}
		float currNorm = normV[v].norm;
		if(currNorm < minNorm) {
			minNorm = currNorm;
			minV = v;
		}
	}

	int numSpanned = normV[minV].numGroups; //num of groups spanned by intermediate v

	groupsSpanned[0] = minV;
	groupsSpanned[1] = numSpanned;

	#pragma omp parallel for
	for(int i = 0; i < numSpanned; i++) 
		groupsSpanned[2 + i] = *(normV[minV].groupss + i); //copy groups spaneed to an array

	for(int i = 0; i < V; i++) { //deallocate memory
		free(normV[i].groupss);
	}
	free(normV);	
}

void partialStarLoop(int src, int *groupIds, int * vertToGroup, int numGroups, int remGroups, int V, int * rTOv, struct InterInfo* normV) {
	for(int v = 0; v < V; v++) { //for each V
		float normm;

		if(v == src)//if root
			continue;

		normm = MAX_NORM + 1.0;
		int minJ = remGroups;

		if(remGroups > 1) {
			bubble_sort2(groupIds, remGroups);
			sort(src, v, groupIds, vertToGroup, numGroups, remGroups); //sort groups
		}
		
		for(int j = 1; j <= remGroups; j++) { //find j that minimizes the norm of current v
			float curr = norm(src,v,j,rTOv,numGroups,vertToGroup,groupIds,V,remGroups);
			if(curr < normm) {
				normm = curr;
				minJ = j;
			}
		}

		normV[v].interm = v; //save into struct
		normV[v].norm = normm;
		normV[v].numGroups = minJ;
		#pragma omp parallel for
		for(int i = 0; i < minJ; i++)  //copy the group spanned by v into an array
			*(normV[v].groupss + i) = groupIds[i];
	}
}

float norm(int src, int v, int j, int* rTOv, int numGroups, int* vertToGroup, int* groupIds, int V, int remGroups) {
	int sum1 = 0, sum2 = 0;
	
	#pragma omp parallel for
	for(int i = 0; i < j; i++) {
		sum1 += vertToGroup[v * numGroups + groupIds[i]];
		sum2 += vertToGroup[src * numGroups + groupIds[i]];
	}
	float normm;

	if(sum2 == 0) {
		normm = MAX_NORM;
	} else {
		normm = ((float) rTOv[v] + (float) sum1) / (float) sum2;
	}
	return normm;
}

void bubble_sort(float* v, int* groupIds, int n)
{
  int c, d;
  int t1;  
  float t2;
  for (c = 0 ; c < ( n - 1 ); c++) {
    for (d = 0 ; d < n - c - 1; d++) {
      if (v[d] > v[d+1]) {
        /* Swapping */
        t2 = v[d];
        v[d] = v[d+1];
        v[d+1] = t2;
        
        t1= groupIds[d];
        groupIds[d]   = groupIds[d+1];
        groupIds[d+1] = t1;  
      }
    }
  }
}

//this just sorts one array
void bubble_sort2(int * groupIds, int n)
{
  int c, d;
  int t1;  
  //float t2;
  for (c = 0 ; c < ( n - 1 ); c++) {
    for (d = 0 ; d < n - c - 1; d++) {
      if (groupIds[d] > groupIds[d+1]) {
        /* Swapping */
        
        t1= groupIds[d];
        groupIds[d]   = groupIds[d+1];
        groupIds[d+1] = t1;  
      }
    }
  }
}

void sort(int src, int interm, int* groupIds, int* vertToGroup, int numGroups, int remGroups) {
	float * arr = (float *) malloc(sizeof(float) * remGroups); //stores cost(r,Ni)/cost(v,Ni) for each group
	if(arr == NULL) {
		return;
	}
	for(int i = 0; i < remGroups; i++) {
		if(vertToGroup[src * numGroups + i] == 0) {
			arr[i] = (float) INT_MAX;
			continue;
		}	
		arr[i] = ((float) vertToGroup[interm * numGroups + groupIds[i]]) / ((float) vertToGroup[src * numGroups + groupIds[i]]);		
	}
	

	bubble_sort(arr, groupIds, remGroups); //bubble sort

	free(arr);
}


int isAvailable(int v, int * intermSet) { //checks if current vertex is not already part of the solution
	for(int i = 0; i < intermSet[0]; i++)
		if(v == intermSet[i+1])
			return 0;
	return 1;
}
