#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include "FSS_Par.cu"

#define BLOCOS 30
#define THREADS 30

// to execute the algorithm
// nvcc -o a main.cu
// ./a 0
// the number can be: 0 = sphere | 1 = Rosenbrock | 2 = Rastrigin | 3 = Griewank

int main( int argc, char* argv[] )
{
	//---------------Initializing variables---------------
	clock_t begin = clock();
	int number_of_fish = BLOCOS;
	int function = atoi(argv[1]);
	double best = 0;
	int interacoes = 5000;
	//---------------TESTS----------------
	int* a;
	a = (int *)malloc(BLOCOS * sizeof(int));
	for(int i=0;i<number_of_fish;i++){
		a[i] = i;
	}
	int* peixes;
	cudaMalloc((void**) &peixes, BLOCOS *sizeof(int));
	cudaMemcpy(peixes, a, BLOCOS*sizeof(int), cudaMemcpyHostToDevice);
	FSS_Par fssAlgor(function, interacoes);
	for(int i=0;i<interacoes;i++){
		runFss(fssAlgor,peixes, number_of_fish, function);
		cudaDeviceSynchronize();
	}

	best = fssAlgor.getBest();
	free(a);
	cudaFree(peixes);

	clock_t end = clock();
	double dif = (double)(end - begin) / CLOCKS_PER_SEC;
  	printf ("The execution took %f seconds.\nThe best result was: %f\n", dif, -best );
	return 0;
}

