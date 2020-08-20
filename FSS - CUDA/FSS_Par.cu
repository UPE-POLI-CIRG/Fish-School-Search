#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <cstdlib>
#include <time.h>
#include <malloc.h>
#include <curand.h>
#include <curand_kernel.h>

// EACH BLOCK = 1 FISH || EACH THREAD = 1 DIMENSION
#define BLOCOS 30
#define THREADS 30
#define STEPINIC 0.1
#define STEPFINAL 0.0001
#define PI 3.14159265359

using namespace std;

class FSS_Par {

private:

	double* best;
	double* pMax;
	double* pMin;
	double* posMax;
	double* posMin;
	
	int* interacoes;
	int* inter;
	int* peixes;
	double* stepHost;
	double* step;
	double* posicoes;
	double* peso;
	double* pesoTotal;
	double* novaPosic_cuda;
	double* dife_cuda;

public:
	//-------- PARA FACILITAR USO NA FUNCAO GLOBAL -> TO DO: ADD gets e sets---------
	double* vectorI;
	double* div;
	double* diferencaFitness;
	int* nPeixes;
	double* deslocamento;
	double* fitness;
	double* best_cuda;
	double* maiorDif;
	double* sumBar;
	double* peso_cuda;
	double* bari;	
	double* posicoes_cuda;
	bool* mudPeso;
	double* pesoTotal_cuda;
	double* dist;
	//------------------------------------------------------
	FSS_Par(int funcao, int interac);
	void memoryFree();
	void inicializar();
	double getBest();
	__device__ double calFitness(double* posi, int peixe, int funcao);
	__device__ void feeding(int* peixesAtivos);
	__device__ void movInd(unsigned int seed, int* peixesAtivos, int funcao);
	__device__ void movColInst(int* peixesAtivos);
	__device__ void movColVol(unsigned int seed, int* peixesAtivos);
	__device__ void attStep();
	__device__ void printBest();

	
};

double FSS_Par::getBest(){
	cudaMemcpy(best, best_cuda , sizeof(double), cudaMemcpyDeviceToHost);
	return best[0];
}

__device__ void FSS_Par::printBest(){
	printf("best = %f \n", -best_cuda[0]);
}

void FSS_Par::inicializar(){

	double pos_ini;
	for(int i=0;i<(BLOCOS*THREADS);i++){
		pos_ini = (double)(rand() % 10000)/10000;
		pos_ini *= pMax[0]/2;
		pos_ini += pMax[0]/2;
		posicoes[i] = pos_ini;
	}
	
}

__device__ double FSS_Par::calFitness(double* posi, int peixe, int funcao){
	
	double fit = 0;
	double aux=0;
	double aux2=1;
	switch(funcao){
		case 0: // Sphere
			for(int i=0;i<THREADS;i++){
				fit+= posi[peixe*THREADS + i] * posi[peixe*THREADS + i];
			}
			return -fit;
		case 1: // Rosenbrock
			for(int i=0;i<THREADS-1;i++){
				aux = posi[peixe*THREADS + i+1] - powf(posi[peixe*THREADS + i],2);
				aux = 100*powf(aux,2);
				aux += powf(1-posi[peixe*THREADS + i],2);
				fit+= aux;
			}
			return -fit;
		case 2: // Rastrigin
			for(int i=0;i<THREADS;i++){
				aux = powf(posi[peixe*THREADS + i],2);
				aux -= 10*cos(PI*2*posi[peixe*THREADS + i]);
				fit+= aux;
			}
			fit +=10*THREADS;
			return -fit;
		case 3: // Griewank
			fit +=1;
			for(int i=1;i<THREADS+1;i++){
				aux += powf(posi[peixe*THREADS + i-1],2)/4000;
				aux2 *= cos(posi[peixe*THREADS + i-1]/sqrtf(i));
				
			}
			fit += aux;
			fit -= aux2;
			return -fit;
		default: 
			printf("FUNCAO ESCOLHIDA NAO VALIDA");
			return 0;
	}
	
	
}


__device__ void FSS_Par::feeding(int* peixesAtivos){ // cada bloco é um peixe 
	
	int index = threadIdx.x;
	
	if(maiorDif[0]!=0){
		peso_cuda[peixesAtivos[index]] += (diferencaFitness[peixesAtivos[index]]/maiorDif[0]);
	}	
	
	
}

__device__ void FSS_Par::attStep(){  // ok
	
	step[0] -= ((posMax[0]-posMin[0])*STEPINIC - (posMax[0]-posMin[0])*STEPFINAL)/inter[0];
	
}

__device__ void FSS_Par::movInd(unsigned int seed, int* peixesAtivos, int funcao){  // ok

	int index = threadIdx.x;
	int block = peixesAtivos[blockIdx.x];
	
	curandState_t state;
	curand_init(seed + index, 0,0, &state);

	if(index==0){
		fitness[block] = calFitness(posicoes_cuda, block, funcao);
	}
	double novoFitness = 0;	
	double aleat=0;

	aleat = curand(&state)%2;
	if(aleat==0){
		aleat=-1;	
	}	
	novaPosic_cuda[block*THREADS + index] = posicoes_cuda[block*THREADS + index] + aleat*step[0];

	if(novaPosic_cuda[block*THREADS + index]>posMax[0]){
		novaPosic_cuda[block*THREADS + index]=posMax[0];	
	}else if(novaPosic_cuda[block*THREADS + index]<posMin[0]){
		novaPosic_cuda[block*THREADS + index]=posMin[0];
	}
	
	__syncthreads();
	
	if(index==0){
		novoFitness = calFitness(novaPosic_cuda, block, funcao);
		if(novoFitness>fitness[block]){
			diferencaFitness[block] = novoFitness - fitness[block];
			fitness[block] = novoFitness;
		}else{
			diferencaFitness[block]=0;
		}
	}
	
	__syncthreads();
	
	if(diferencaFitness[block]==0){
		deslocamento[block*THREADS + index] = 0;
	}else{
		deslocamento[block*THREADS + index] = novaPosic_cuda[block*THREADS + index]-posicoes_cuda[block*THREADS + index];
		posicoes_cuda[block*THREADS + index] = novaPosic_cuda[block*THREADS + index];
	}	
}

__device__ void FSS_Par::movColInst(int* peixesAtivos){  // ok
	
	int index = threadIdx.x;
	int block = peixesAtivos[blockIdx.x];

	posicoes_cuda[block*THREADS + index] += vectorI[index];
	
	if(posicoes_cuda[block*THREADS + index]>posMax[0]){
		posicoes_cuda[block*THREADS + index]=posMax[0];	
	}else if(posicoes_cuda[block*THREADS + index]<posMin[0]){
		posicoes_cuda[block*THREADS + index]=posMin[0];
	}	
	
}

__device__ void FSS_Par::movColVol(unsigned int seed, int* peixesAtivos){  // ok
	
	int index = threadIdx.x;
	int block = peixesAtivos[blockIdx.x];
	curandState_t state;
	curand_init(seed + index, 0,0, &state);
	//double aleat = ((double)(curand(&state)%10000))/10000; 
	int aleat = curand(&state)%2;
	
	__syncthreads();
	dife_cuda[block*THREADS + index] = posicoes_cuda[block*THREADS + index] - bari[index];
	
	
	if(mudPeso[0]){
		if(dist[block * THREADS]!=0){
			posicoes_cuda[block*THREADS + index] = posicoes_cuda[block*THREADS + index] + 
				2*step[0]*aleat*dife_cuda[block*THREADS + index]/dist[block * THREADS];
		}
		if(posicoes_cuda[block*THREADS + index]>posMax[0]){
			posicoes_cuda[block*THREADS + index]=posMax[0];	
		}else if(posicoes_cuda[block*THREADS + index]<posMin[0]){
			posicoes_cuda[block*THREADS + index]=posMin[0];
		}
	}else{
		if(dist[block * THREADS]!=0){
			posicoes_cuda[block*THREADS + index] = posicoes_cuda[block*THREADS + index] -
				2*step[0]*aleat*dife_cuda[block*THREADS + index]/dist[block * THREADS];
		}
		if(posicoes_cuda[block*THREADS + index]>posMax[0]){
			posicoes_cuda[block*THREADS + index]=posMax[0];	
		}else if(posicoes_cuda[block*THREADS + index]<posMin[0]){
			posicoes_cuda[block*THREADS + index]=posMin[0];
		}
	}
}

FSS_Par::FSS_Par(int funcao, int interac){

	//Inicializando Variaveis----------
	pMax = (double *)malloc(1 * sizeof(double));
	pMin = (double *)malloc(1 * sizeof(double));
	interacoes = (int *)malloc(sizeof(int));
	stepHost = (double *)malloc(sizeof(double));
	pMax[0] = 0;
	pMin[0] = 0;
	interacoes[0] = interac;
	//Posições max e min de cada função-------------
	switch(funcao){
		case 0:
			pMax[0] = 100;
			pMin[0] = -100;
			break;
		case 1:
			pMax[0] = 30;
			pMin[0] = -30;
			break;
		case 2:
			pMax[0] = 5.12;
			pMin[0] = -5.12;
			break;
		case 3:
			pMax[0] = 600;
			pMin[0] = -600;
			break;
		default:
			cout << "ERRO FUNCAO ESCOLHIDA NAO VALIDA" << endl;
	}
	stepHost[0] = (pMax[0]-pMin[0])*STEPINIC;
	//---------------------------------------------

	//Para Random----------------------------------
	srand (time(NULL));
	//---------------------------------------------

	//Alocando Ponteiros------------------------------------------------------------------
	posicoes = (double *)malloc(THREADS * BLOCOS * sizeof(double));
	peso = (double *)malloc(BLOCOS * sizeof(double));
	pesoTotal = (double *)malloc(sizeof(double));
	best = (double *)malloc(sizeof(double));
	cudaMalloc((void**) &div, 1 *sizeof(double));
	cudaMalloc((void**) &mudPeso, 1 *sizeof(bool));
	cudaMalloc((void**) &sumBar, 1 *sizeof(double));
	cudaMalloc((void**) &nPeixes, 1 *sizeof(int));
	cudaMalloc((void**) &peixes, BLOCOS *sizeof(int));
	cudaMalloc((void**) &posMax, 1 *sizeof(double));
	cudaMalloc((void**) &posMin, 1 *sizeof(double));	
	cudaMalloc((void**) &step, 1 *sizeof(double));
	cudaMalloc((void**) &dist, THREADS*BLOCOS *sizeof(double));
	cudaMalloc((void**) &inter, 1 *sizeof(int));	
	cudaMalloc((void**) &posicoes_cuda, THREADS * BLOCOS *sizeof(double));
	cudaMalloc((void**) &diferencaFitness, BLOCOS * sizeof(double));
	cudaMalloc((void**) &deslocamento, THREADS * BLOCOS * sizeof(double));
	cudaMalloc((void**) &peso_cuda, BLOCOS * sizeof(double));
	cudaMalloc((void**) &maiorDif, 1 *sizeof(double));
	cudaMalloc((void**) &vectorI, BLOCOS*THREADS *sizeof(double));
	cudaMalloc((void**) &bari, BLOCOS*THREADS *sizeof(double));
	cudaMalloc((void**) &pesoTotal_cuda, BLOCOS*sizeof(double));
	cudaMalloc((void**) &best_cuda, sizeof(double));
	cudaMalloc((void**) &fitness, BLOCOS*sizeof(double));
	cudaMalloc((void**) &novaPosic_cuda, THREADS*BLOCOS*sizeof(double));
	cudaMalloc((void**) &dife_cuda, THREADS*BLOCOS*sizeof(double));
	//------------------------------------------------------------------------------------

	//Inicializando FSS--------------------------------
	inicializar();
	for(int i=0;i<BLOCOS;i++){peso[i] = 1.0;}
	pesoTotal[0] = BLOCOS;
	best[0] = -9999999;
	//-------------------------------------------------

	//Passando dados do host para device----------------------------------------------------------------------------
	cudaMemcpy(posicoes_cuda, posicoes , THREADS * BLOCOS  *sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(peso_cuda, peso , BLOCOS  *sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(pesoTotal_cuda, pesoTotal , sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(best_cuda, best , sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(posMax, pMax , sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(posMin, pMin , sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(inter, interacoes , sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(step, stepHost , sizeof(double), cudaMemcpyHostToDevice);
	//--------------------------------------------------------------------------------------------------------------

}

void FSS_Par::memoryFree(){

	free(posicoes);
	free(peso);
	free(pesoTotal);
	free(best);
	cudaFree(posicoes_cuda);
	cudaFree(diferencaFitness);
	cudaFree(deslocamento);	
	cudaFree(peso_cuda);
	cudaFree(maiorDif);	
	cudaFree(vectorI);
	cudaFree(bari);
	cudaFree(pesoTotal_cuda);
	cudaFree(best_cuda);
	cudaFree(fitness);
	cudaFree(novaPosic_cuda);
	cudaFree(dife_cuda);
	cudaFree(nPeixes);
	cudaFree(peixes);
	cudaFree(posMax);
	cudaFree(posMin);
	cudaFree(step);
	cudaFree(inter);
	cudaFree(div);
}


__device__ double atomicAddA(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}



__global__ void direcIAux(FSS_Par fss, int* peixesAtivos){

	int index = threadIdx.x;
	atomicAddA(fss.div, fss.diferencaFitness[peixesAtivos[index]]); // para cada peixe
}

__global__ void direcIAux2(FSS_Par fss, int* peixesAtivos){

	int index = threadIdx.x;

	if(fss.div[0]!=0){	
		fss.vectorI[index] =  fss.vectorI[peixesAtivos[0]*THREADS + index]/fss.div[0];
	}	
	
}
__global__ void direcI(FSS_Par fss, int* peixesAtivos){

	int aux = fss.nPeixes[0];
	int index = threadIdx.x;
	int block = blockIdx.x;
	int index2 = -1;
	fss.vectorI[peixesAtivos[index]*THREADS + block] = fss.diferencaFitness[peixesAtivos[index]]*fss.deslocamento[peixesAtivos[index]*THREADS + block];
	
	__syncthreads();
	while(aux!=1){
		int maxIndex = aux/2;
		if(aux % 2==0){
			index2 = index + maxIndex;
		}else{
			index2 = index + maxIndex+1;
		}
		if(index>=0 && index<maxIndex){
			fss.vectorI[peixesAtivos[index]*THREADS + block] += fss.vectorI[peixesAtivos[index2]*THREADS + block];
		}
		aux = aux % 2 == 0 ? aux / 2 : (aux / 2) + 1;
		__syncthreads();
	}	
}

__global__ void baricentroAux(FSS_Par fss, int* peixesAtivos){

	int index = threadIdx.x;
	atomicAddA(fss.sumBar, fss.peso_cuda[peixesAtivos[index]]); // para cada peixe
}

__global__ void baricentro(FSS_Par fss, int* peixesAtivos){ 
		
	
	int aux = fss.nPeixes[0];
	int index = threadIdx.x;
	int block = blockIdx.x;
	int index2 = -1;
	fss.bari[peixesAtivos[index]] = 0;
	fss.bari[peixesAtivos[index]*THREADS + block] = fss.peso_cuda[peixesAtivos[index]]*fss.posicoes_cuda[peixesAtivos[index]*THREADS + block];
	
	__syncthreads();
	while(aux!=1){
		int maxIndex = aux/2;
		if(aux % 2==0){
			index2 = index + maxIndex;
		}else{
			index2 = index + maxIndex+1;
		}
		if(index>=0 && index<maxIndex){
			fss.bari[peixesAtivos[index]*THREADS + block] += fss.bari[peixesAtivos[index2]*THREADS + block];
		}
		aux = aux % 2 == 0 ? aux / 2 : (aux / 2) + 1;
		__syncthreads();
	}
}

__global__ void baricentroAux2(FSS_Par fss, int* peixesAtivos){

	int index = threadIdx.x;

	if(fss.sumBar[0]!=0){
		fss.bari[index] =  fss.bari[peixesAtivos[0]*THREADS + index]/fss.sumBar[0];
		//printf("bari %d = %f \n", index, fss.bari[index]);	
	}	
	
}

__global__ void maiorPeso(FSS_Par fss,int* peixesAtivos){ // true se o peso diminuiu
	
	int aux = fss.nPeixes[0];
	int index1 = threadIdx.x;
	int index2 = -1;	
	
	int peixe = peixesAtivos[index1]; // para restaurar a lista de peixes ativos dps de modifica-la
	double pesoAnterior = fss.pesoTotal_cuda[0];
	fss.pesoTotal_cuda[peixesAtivos[index1]] = fss.peso_cuda[peixesAtivos[index1]];	
		
	__syncthreads();
	while(aux!=1){
		int maxIndex = aux/2;
		if(aux % 2==0){
			index2 = index1 + maxIndex;
		}else{
			index2 = index1 + maxIndex+1;
		}
		if(index1>=0 && index1<maxIndex){
			fss.pesoTotal_cuda[peixesAtivos[index1]] += fss.pesoTotal_cuda[peixesAtivos[index2]];
				
		}
		aux = aux % 2 == 0 ? aux / 2 : (aux / 2) + 1;
		__syncthreads();
	}
	__syncthreads();
	if(index1==0){
		fss.pesoTotal_cuda[0] = fss.pesoTotal_cuda[peixesAtivos[index1]];
		if(fss.pesoTotal_cuda[0]>pesoAnterior){
			fss.mudPeso[0] = false;
		}else{
			fss.mudPeso[0] = true;	
		}
		//printf("mud peso = %d ", fss.mudPeso[0]);
	}

	peixesAtivos[index1] = peixe;
}

__global__ void bestResult(FSS_Par fss, int* peixesAtivos){  

	int aux = fss.nPeixes[0];
	int index1 = threadIdx.x;
	int index2 = -1;	
	
	int peixe = peixesAtivos[index1]; // para restaurar a lista de peixes ativos dps de modifica-la
	
	__syncthreads();
	while(aux!=1){
		int maxIndex = aux/2;
		if(aux % 2==0){
			index2 = index1 + maxIndex;
		}else{
			index2 = index1 + maxIndex+1;
		}
		if(index1>=0 && index1<maxIndex){
			double fitness1 = fss.fitness[peixesAtivos[index1]];
			double fitness2 = fss.fitness[peixesAtivos[index2]];
			peixesAtivos[index1] = fitness1 > fitness2 ? peixesAtivos[index1] : peixesAtivos[index2];
			
		}
		aux = aux % 2 == 0 ? aux / 2 : (aux / 2) + 1;
		__syncthreads();
	}
	__syncthreads();
	if(index1==0){
		if(fss.fitness[peixesAtivos[0]] > fss.best_cuda[0]){
			fss.best_cuda[0] = fss.fitness[peixesAtivos[0]];
		}
	}

	peixesAtivos[index1] = peixe;
}

__global__ void maiorDifFit(FSS_Par fss, int* peixesAtivos){
	
	int aux = fss.nPeixes[0];
	int index1 = threadIdx.x;
	int index2 = -1;	
	
	int peixe = peixesAtivos[index1]; // para restaurar a lista de peixes ativos dps de modifica-la
	
	__syncthreads();
	while(aux!=1){
		int maxIndex = aux/2;
		if(aux % 2==0){
			index2 = index1 + maxIndex;
		}else{
			index2 = index1 + maxIndex+1;
		}
		if(index1>=0 && index1<maxIndex){
			double difefitness1 = fss.diferencaFitness[peixesAtivos[index1]];
			double difefitness2 = fss.diferencaFitness[peixesAtivos[index2]];
			peixesAtivos[index1] = difefitness1 > difefitness2 ? peixesAtivos[index1] : peixesAtivos[index2];
				
		}
		aux = aux % 2 == 0 ? aux / 2 : (aux / 2) + 1;
		__syncthreads();
	}
	__syncthreads();
	if(index1==0){
		fss.maiorDif[0] = fss.diferencaFitness[peixesAtivos[0]];
		//printf("maior dif = %f \n",fss.maiorDif[0]);
	}

	peixesAtivos[index1] = peixe;		
}



__global__ void nPeixes_glob(FSS_Par fss, int numeroPeixes){

	fss.nPeixes[0] = numeroPeixes;
}

__global__ void movInd_glob(FSS_Par fss, unsigned int seed, int* peixesAtivos, int funcao){
	
	fss.movInd(seed,peixesAtivos, funcao);
}

__global__ void movInst_glob(FSS_Par fss, int* peixesAtivos){
	
	fss.movColInst(peixesAtivos);
}

__global__ void movVol_glob(FSS_Par fss, unsigned int seed, int* peixesAtivos){
	
	fss.movColVol(seed,peixesAtivos);
}

__global__ void attStep_glob(FSS_Par fss, int* peixesAtivos){
	
	fss.attStep();
	//fss.printPo(peixesAtivos);
}

__global__ void feeding_glob(FSS_Par fss, int* peixesAtivos){
	fss.feeding(peixesAtivos);
}

__global__ void distancia(FSS_Par fss, int* peixesAtivos){
	
	int index = threadIdx.x;
	int block = blockIdx.x;
	int aux = THREADS;
	
	int index2=-1;
	fss.dist[peixesAtivos[block] * THREADS + index] = powf(fss.posicoes_cuda[peixesAtivos[block] * THREADS + index] - fss.bari[index],2);	
	__syncthreads();
	
	while(aux!=1){
		int maxIndex = aux/2;
		if(aux % 2==0){
			index2 = index + maxIndex;
		}else{
			index2 = index + maxIndex+1;
		}
		if(index>=0 && index<maxIndex){
			 fss.dist[peixesAtivos[block] * THREADS + index] += fss.dist[peixesAtivos[block] * THREADS + index2];
		}
		aux = aux % 2 == 0 ? aux / 2 : (aux / 2) + 1;
		__syncthreads();
	}
	__syncthreads();
	if(index==0){
		fss.dist[peixesAtivos[block] * THREADS] = sqrtf(fss.dist[peixesAtivos[block] * THREADS]);
		//printf("%f \n", fss.dist[peixesAtivos[block] * THREADS]);
	}
}

void runFss(FSS_Par fss, int* peixesAtivos, int numeroPeixes, int funcao){

	//--------------SETUP INICIAL---------------------------------------------
	struct timespec rawtime;
	unsigned int seed;
	clock_gettime(CLOCK_MONOTONIC_RAW, &rawtime);
	seed = rawtime.tv_nsec;
	nPeixes_glob<<<1,1>>>(fss, numeroPeixes);
	maiorPeso<<<1,numeroPeixes>>>(fss, peixesAtivos); // como os peixes usados podem mudar eh preciso verificar o peso dos peixes que serao usados nessa interacao
	//--------------MOVIM INDIV-----------------------------------------------
	movInd_glob<<<numeroPeixes,THREADS>>>(fss, seed,peixesAtivos, funcao);
	//--------------VECTOR I--------------------------------------------------
	direcIAux<<<1, numeroPeixes>>>(fss, peixesAtivos);
	direcI<<<THREADS, numeroPeixes>>>(fss, peixesAtivos); // tem um bloco para cada thread(dimensao) pq as dimensoes independem uma da outra
	direcIAux2<<<1, THREADS>>>(fss, peixesAtivos);
	//----------Maior diferenca entre os fitness------------------------------
	maiorDifFit<<<1,numeroPeixes>>>(fss, peixesAtivos);
	//-----------------FEEDING------------------------------------------------
	feeding_glob<<<1,numeroPeixes>>>(fss, peixesAtivos);
	//--------------MOVIM INSTN-----------------------------------------------
	movInst_glob<<<numeroPeixes,THREADS>>>(fss,peixesAtivos);
	//--------------BARICENTRO------------------------------------------------
	baricentroAux<<<1, numeroPeixes>>>(fss, peixesAtivos);
	baricentro<<<THREADS, numeroPeixes>>>(fss, peixesAtivos);
	baricentroAux2<<<1, THREADS>>>(fss, peixesAtivos);

	maiorPeso<<<1,numeroPeixes>>>(fss, peixesAtivos);
	distancia<<<numeroPeixes, THREADS>>>(fss, peixesAtivos);
	movVol_glob<<<numeroPeixes,THREADS>>>(fss, seed,peixesAtivos);
	//--------------SETUP FINAL------------------------------------------------
	bestResult<<<1,numeroPeixes>>>(fss, peixesAtivos);
	attStep_glob<<<1,1>>>(fss, peixesAtivos);
}
