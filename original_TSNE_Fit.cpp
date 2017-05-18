/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */



#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include "vptree.h"
#include "sptree.h"
#include "tsne.h"

#include <boost/thread.hpp>
#include <boost/random.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

using namespace std;

char input_file[1000], output_file[1000], log_file[1000];

typedef float real;

struct ClassVertex {
	double degree;
	char *name;
};

struct ClassVertex *vertex;

FILE *flog;

SPTreeBH* tree;

double *mmmean;

long long nnnum_vertices, vvvector_dim, nnnum_threads = 1;
double *gggglobal_negf, gggglobal_theta, *gggglobal_sumq, *buff;

double *gggglobal_gains, *gggglobal_dY, *gggglobal_uY, *gggglobal_Y;
double *gggglobal_eta, *gggglobal_momentum;
long long *gggglobal_N, *gggglobal_no_dims;
float build_tree = 0;
void *updateGradientThread(void *_id)
{
	long long id = (long long)_id;
	long long lo = id * *gggglobal_N / nnnum_threads;
	long long hi = (id + 1) * *gggglobal_N / nnnum_threads;
//	printf("%lld %lld %lld\n", id, lo, hi);
	for (long long i = lo * *gggglobal_no_dims; i < hi * *gggglobal_no_dims; ++i)
	{
		//update gains
		gggglobal_gains[i] = (sign(gggglobal_dY[i]) != sign(gggglobal_uY[i])) ? (gggglobal_gains[i] + .2) : (gggglobal_gains[i] * .8);
		if (gggglobal_gains[i] < .01) gggglobal_gains[i] = .01;
		//update gradient
		gggglobal_uY[i] = *gggglobal_momentum * gggglobal_uY[i] - *gggglobal_eta * gggglobal_gains[i] * gggglobal_dY[i];
		gggglobal_Y[i] = gggglobal_Y[i] + gggglobal_uY[i];
	}

	return NULL;
}
// Perform t-SNE
void TSNE::run(long long N, double* Y, long long no_dims, double perplexity, double theta, unsigned long long* row_P, unsigned long long* col_P, double* val_P) {
    
    // Determine whether we are using an exact algorithm
    if(N - 1 < 3 * perplexity) { printf("Perplexity too large for the number of data points!\n"); exit(1); }
   
	printf("Using no_dims = %lld, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);
	fprintf(flog, "Using no_dims = %lld, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);
    
    // Set learning parameters
    float total_time = .0;

    clock_t start, end;
	int max_iter = 1000, stop_lying_iter = 250, mom_switch_iter = 250;
	double momentum = .5, final_momentum = .8;
	double eta = 200.0;
	gggglobal_momentum = &momentum;
	gggglobal_eta = &eta;
	gggglobal_N = &N;
	gggglobal_no_dims = &no_dims;

    // Allocate some memory
	double* dY = (double*)malloc(N * no_dims * sizeof(double)); gggglobal_dY = dY;
	double* uY = (double*)malloc(N * no_dims * sizeof(double)); gggglobal_uY = uY;
	double* gains = (double*)malloc(N * no_dims * sizeof(double)); gggglobal_gains = gains;
	gggglobal_Y = Y;

    if(dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (long long i = 0; i < N * no_dims; i++)    uY[i] = .0;
	for (long long i = 0; i < N * no_dims; i++) gains[i] = 1.0;
    
    
    // Lie about the P-values
	for (long long i = 0; i < row_P[N]; i++) val_P[i] *= 12.0;

	// Initialize solution (randomly)
	for (long long i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
	
    start = clock();
	for(int iter = 0; iter < max_iter; iter++) {

        // Compute (approximate) gradient
        computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta);


//		printf("===\n");
		boost::thread *pt = new boost::thread[nnnum_threads];
		for (long long i = 0; i < nnnum_threads; ++i) pt[i] = boost::thread(updateGradientThread, (void*)i);
		for (long long i = 0; i < nnnum_threads; ++i) pt[i].join();
		delete[] pt;
		/*
        // Update gains
		for (long long i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
		for (long long i = 0; i < N * no_dims; i++) if (gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
		for (long long i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];
        */
        // Make solution zero-mean
//		printf("===\n");
		zeroMean(Y, N, no_dims);
        
        // Stop lying about the P-values after a while, and switch momentum
        if(iter == stop_lying_iter) {
			for (long long i = 0; i < row_P[N]; i++) val_P[i] /= 12.0;
        }
        if(iter == mom_switch_iter) momentum = final_momentum;
        
        // Print out progress
        if(iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
            end = clock();
            double C = .0;
            
//			C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);  // doing approximate computation here!
            
			if (iter == 0)
			{
				printf("Iteration %d: error is %f\n", iter + 1, C);
				fprintf(flog, "Iteration %d: error is %f\n", iter + 1, C);
			}
            else {
                total_time += (float) (end - start) / CLOCKS_PER_SEC;
                printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float) (end - start) / CLOCKS_PER_SEC);
				fprintf(flog, "Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float)(end - start) / CLOCKS_PER_SEC);
            }
			start = clock();
        }
    }
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;
    
    // Clean up memory
    free(dY);
    free(uY);
    free(gains);
   
        free(row_P); row_P = NULL;
        free(col_P); col_P = NULL;
        free(val_P); val_P = NULL;
   
    printf("Fitting performed in %4.2f seconds.\n", total_time);
	printf("Time of building BH tree is %4.2f seconds.\n", build_tree);
	fprintf(flog, "Fitting performed in %4.2f seconds.\n", total_time);
	fprintf(flog, "Time of building BH tree is %4.2f seconds.\n", build_tree);
}

void *computeNonEdgeForcesThread(void *_id)
{
	long long id = (long long)_id;
	long long lo = id * nnnum_vertices / nnnum_threads;
	long long hi = (id + 1) * nnnum_vertices / nnnum_threads;

	long long i;
//	double sum_Q;
	gggglobal_sumq[id] = 0;
	for (i = lo; i < hi; ++i)
	{
		tree->computeNonEdgeForces(i, gggglobal_theta, gggglobal_negf + i * vvvector_dim, &gggglobal_sumq[id], &buff[id * vvvector_dim]);
//		gggglobal_sumq[id] += sum_Q;
	}

	return NULL;
}

// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(unsigned long long* inp_row_P, unsigned long long* inp_col_P, double* inp_val_P, double* Y, long long N, long long D, double* dC, double theta)
{
//	printf("Start computing gradient ... \n");
    // Construct space-partitioning tree on current map
	clock_t ss = clock();
    tree = new SPTree(D, Y, N);
	build_tree += (float)(clock() - ss) / CLOCKS_PER_SEC;
//	printf("Finished building tree\n");
    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    double* pos_f = (double*) calloc(N * D, sizeof(double));
    double* neg_f = (double*) calloc(N * D, sizeof(double));
    if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f, nnnum_threads);
	gggglobal_negf = neg_f;
	gggglobal_theta = theta;

//	for (long long n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);
    
	boost::thread *pt = new boost::thread[nnnum_threads];
	for (long long i = 0; i < nnnum_threads; ++i) pt[i] = boost::thread(computeNonEdgeForcesThread, (void*)i);
	for (long long i = 0; i < nnnum_threads; ++i) pt[i].join();
	delete[] pt;
	for (long long i = 0; i < nnnum_threads; ++i)
		sum_Q += gggglobal_sumq[i];

    // Compute final t-SNE gradient
	for (long long i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }
    free(pos_f);
    free(neg_f);
//	printf("Finished computing\n");
    delete tree;
//	printf("Finished deleting tree\n");
}

// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(unsigned long long* row_P, unsigned long long* col_P, double* val_P, double* Y, long long N, long long D, double theta)
{
    
    // Get estimate of normalization term
    tree = new SPTree(D, Y, N);
    double* buff = (double*) calloc(D * N, sizeof(double));
    double sum_Q = .0;

	gggglobal_negf = buff;
	boost::thread *pt = new boost::thread[nnnum_threads];
	for (long long i = 0; i < nnnum_threads; ++i) pt[i] = boost::thread(computeNonEdgeForcesThread, (void*)i);
	for (long long i = 0; i < nnnum_threads; ++i) pt[i].join();
	delete[] pt;
	for (long long i = 0; i < nnnum_threads; ++i)
		sum_Q += gggglobal_sumq[i];

//	for (long long n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);
    
    // Loop over all edges to compute t-SNE error
	long long ind1, ind2;
    double C = .0, Q;
	for (long long n = 0; n < N; n++) {
        ind1 = n * D;
		for (long long i = row_P[n]; i < row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * D;
			for (long long d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (long long d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (long long d = 0; d < D; d++) Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }
    
    // Clean up memory
    free(buff);
    delete tree;
    return C;
}


// Makes data zero-mean
void TSNE::zeroMean(double* X, long long N, long long D) {
	
	// Compute data mean
	long long nD = 0;
	for (long long d = 0; d < D; ++d)
		mean[d] = 0;
	for (long long n = 0; n < N; n++) {
		for (long long d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for (long long d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}
	
	// Subtract data mean
    nD = 0;
	for (long long n = 0; n < N; n++) {
		for (long long d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}
}


// Generates a Gaussian random number
double TSNE::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

void Output(long long N, long long D, double *Y)
{
	FILE *fout = fopen(output_file, "wb");

	fprintf(fout, "%lld %lld\n", N, D);

	for (long long i = 0; i < N; ++i)
	{
		fprintf(fout, "%s ", vertex[i].name);
		real real_tmp;
		for (long long j = 0; j < D; ++j)
		{
			real_tmp = Y[i * D + j];
			fwrite(&real_tmp, sizeof(real), 1, fout);
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}

// Function that runs the Barnes-Hut implementation of t-SNE
void Train(long long origN, long long no_dims, double theta, double perplexity, unsigned long long* row_P, unsigned long long* col_P, double* val_P) {
    
    // Define some variables
    int rand_seed = -1;
    TSNE* tsne = new TSNE();
    
    // Read the parameters and the dataset
        
        // Set random seed
        if(rand_seed >= 0) {
            printf("Using random seed: %d\n", rand_seed);
            srand((unsigned int) rand_seed);
        }
        else {
            printf("Using current time as random seed...\n");
            srand(time(NULL));
        }        
        
		// Make dummy landmarks
		long long N = origN;

		// Now fire up the SNE implementation
		double* Y = (double*) malloc(N * no_dims * sizeof(double));
        if(Y == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		tsne->run(N, Y, no_dims, perplexity, theta, row_P, col_P, val_P);		
      
		Output(origN, no_dims, Y);

        // Clean up the memory
		free(Y); Y = NULL;

	delete(tsne);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	long long i;

	strcpy(log_file, "tsne_log.txt");
	double theta = 0.5, perplexity = 50;
	if ((i = ArgPos((char *)"-input", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-log", argc, argv)) > 0) strcpy(log_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) nnnum_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-theta", argc, argv)) > 0) theta = atof(argv[i + 1]);

	flog = fopen(log_file, "wb");

	long long origN, no_dims = 2;
	long long K = (long long)(3 * perplexity); 
	mean = (double*)calloc(no_dims, sizeof(double));
	unsigned long long* row_P;
	unsigned long long* col_P;
	double* val_P;

	FILE *fin = fopen(input_file, "rb");
	printf("Data file : %s\n", input_file);
	fscanf(fin, "%lld", &origN);
	printf("Vertices num\t:\t%lld\n", origN);
	fprintf(flog, "Vertices num\t:\t%lld\n", origN);
	nnnum_vertices = origN;
	vvvector_dim = 2;

	vertex = (struct ClassVertex*)malloc((origN + 10) * sizeof(struct ClassVertex));
	buff = (double*)malloc(nnnum_threads * vvvector_dim * sizeof(double));
	gggglobal_sumq = (double*)malloc(nnnum_threads * sizeof(double));

	char word[1000];
	for (i = 0; i < origN; ++i)
	{
		fscanf(fin, "%s", word);
		vertex[i].name = (char*)malloc((strlen(word) + 1) * sizeof(char));
		strcpy(vertex[i].name, word);
	}

	row_P = (unsigned long long*)malloc((origN + 1) * sizeof(unsigned long long));
	char ch = fgetc(fin);
	fread(row_P, sizeof(unsigned long long), (origN + 1), fin);

	printf("Edges num\t:\t%lld\n", row_P[origN]);
	fprintf(flog, "Edges num\t:\t%lld\n", row_P[origN]);

	col_P = (unsigned long long*)malloc(row_P[origN] * sizeof(unsigned long long));
	val_P = (double*)malloc(row_P[origN] * sizeof(double));

	for (i = 0; i < origN; ++i)
	{
		fread(&col_P[row_P[i]], sizeof(unsigned long long), row_P[i + 1] - row_P[i], fin);
	}
	for (i = 0; i < origN; ++i)
	{
		fread(&val_P[row_P[i]], sizeof(double), row_P[i + 1] - row_P[i], fin);
	}

	fclose(fin);

	/*{//=========================
		int i, j;
		long long nnnum_vertices = origN;
		FILE *fnum = fopen("net_num.txt", "wb");
		FILE *fbin = fopen("net_bin.txt", "wb");

		for (i = 0; i < nnnum_vertices; ++i)
		{
			for (j = row_P[i]; j < row_P[i + 1]; ++j)
			{
				fprintf(fnum, "%s %s ", vertex[i].name, vertex[col_P[j]].name);
				fprintf(fbin, "%s %s ", vertex[i].name, vertex[col_P[j]].name);
				double dbl_tmp = val_P[j];
				dbl_tmp *= 10000000;
				fwrite(&dbl_tmp, sizeof(double), 1, fbin);
				fprintf(fnum, "%.6lf", dbl_tmp);
				fprintf(fbin, "\n");
				fprintf(fnum, "\n");
			}
		}
		fclose(fbin);
		fclose(fnum);
	}*/
	Train(origN, no_dims, theta, perplexity, row_P, col_P, val_P);

	fclose(flog);

//	system("pause");
	return 0;
}

