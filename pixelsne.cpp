/// from PixelSNE: https://github.com/awesome-davian/pixelsne
/// Modified by Ryangwook Ryoo, 2017

#include <math.h>
#include <float.h>
#include <limits.h>

#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include "vptree.h"
#include "pixelsne.h"

#include "ptree.h"
#include "LargeVis.h"

#include <boost/thread.hpp>
using namespace std;


#define BILLION 1000000000L

long long num_insert = 0;

/*Op*/
#define EXP_LUT_RANGE 10
#define EXP_LUT_DIV 1000
double *pexp;
double fexp(double num)
{
	bool flag = false;
	if (num <= -EXP_LUT_RANGE)return pexp[0];
	if (num >= EXP_LUT_RANGE)return pexp[EXP_LUT_DIV - 1];
	return pexp[(int)((num + EXP_LUT_RANGE)*EXP_LUT_DIV / EXP_LUT_RANGE / 2)];
}
double *global2_negf, *global2_theta, *global2_beta, *global2_sumq;
int *global2_iter_cnt, *global2_N, *global2_D;
double *global2_gains, *global2_dY, *global2_uY, *global2_Y;
double *global2_eta, *global2_momentum;
int *global2_no_dims, *global2_skip_cnt, *global2_skip;
bool globalIsSleeping;
int num_threads = 4;
PTree *ptree;
PTree *gradientTree;

PixelSNE::PixelSNE() {
    KNNupdated = false;
    fitting_real_time = 0;
    fitting_cpu_time = 0;
    propagation_cpu_time = 0;
    propagation_real_time = 0;
    isLogging = false;
    isSleeping = false;
    temptime1=0;
    temptime2=0;
    init_real_time=0;
    init_cpu_time=0;
    n_threads = 4;
    skip = NULL;
    stop_lying_iter_num = 250;
}

PixelSNE::~PixelSNE() {
    if (tree != NULL) delete tree;
    if (skip != NULL) free(skip); skip = NULL;
}

void PixelSNE::run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta,
               unsigned int bins, int p_method, int rand_seed, int nthreads, int propagation_num, bool skip_random_init, int n_trees, bool isValidation, int max_iter, int stop_lying_iter, 
               int mom_switch_iter) {
    knn_validation = isValidation; 
    max_iteration = max_iter;
    n_propagations = propagation_num;
    n_threads = nthreads;
    num_threads = nthreads;
    stop_lying_iter_num = stop_lying_iter;
                tempN = N;
    // Set random seed
    if (skip_random_init != true) {
      if(rand_seed >= 0) {
          printf("PixelSNE: Using random seed: %d\n", rand_seed);
          srand((unsigned int) rand_seed);
      } else {
          printf("PixelSNE: Using current time as random seed...\n");
          srand(time(NULL));
      }
    }

    // Determine whether we are using an exact algorithm
    if(N - 1 < 3 * perplexity) { printf("PixelSNE: Perplexity too large for the number of data points!\n"); exit(1); }
    printf("PixelSNE: Using no_dims = %d, perplexity = %f, bins = %d, p_method = %d and theta = %f\n", no_dims, perplexity, bins, p_method, theta);
    exact = (theta == .0) ? true : false;
    
	momentum = .5;
    final_momentum = .8;
	eta = 200.0;

    // Allocate some memory
    dY    = (double*) malloc(N * no_dims * sizeof(double));
    uY    = (double*) malloc(N * no_dims * sizeof(double));
    gains = (double*) malloc(N * no_dims * sizeof(double));
    if(dY == NULL || uY == NULL || gains == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    for(int i = 0; i < N * no_dims; i++)    uY[i] =  .0;

    for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

    // Normalize input data (to prevent numerical problems)
    printf("PixelSNE: Computing input similarities...\n");

    P = NULL;                      
    row_P = NULL;        
    col_P = NULL;
    val_P = NULL;


    if(exact) { 
        // Compute similarities
        printf("PixelSNE: Exact?\n");
    }

    else {  
        start = clock();
        clock_gettime(CLOCK_MONOTONIC, &start_p);

        // Using LargeVis
        if (p_method != 1) {
            printf("PixelSNE: P Method: Construct_KNN\n");

            long long if_embed = 1, out_dim = -1, n_samples = -1, n_negative = -1, n_neighbors = -1;
            long long threadNum = nthreads;
            float alpha = -1, n_gamma = -1;

            p_model = new LargeVis();
            p_model->load_from_data(X, N, D);
            p_model->run(out_dim, threadNum, n_samples, n_propagations, alpha, n_trees, n_negative, n_neighbors, n_gamma, perplexity);
            p_model->get_result(&row_P, &col_P, &val_P);

            start = clock();
            clock_gettime(CLOCK_MONOTONIC, &start_p);

            double sum_P = .0;
            for(int i = 0; i < row_P[N]; i++) {
                sum_P += val_P[i];
            }
            for(int i = 0; i < row_P[N]; i++) {
                val_P[i] /= sum_P;
            }

            double* largeVisRealTime = p_model->get_real_time();
            double* largeVisClockTime = p_model->get_clock_time();
            init_real_time += (largeVisRealTime[0] + largeVisRealTime[1]);
            init_cpu_time += (largeVisClockTime[0] + largeVisClockTime[1]);
        }
        else 
        {
            printf("PixelSNE: P Method: VP Tree\n");

            // Compute asymmetric pairwise input similarities
            computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, (int) (3 * perplexity));

            // Symmetrize input similarities
            symmetrizeMatrix(&row_P, &col_P, &val_P, N);

            double sum_P = .0;
            for(int i = 0; i < row_P[N]; i++) {
                sum_P += val_P[i];
            }
            for(int i = 0; i < row_P[N]; i++) {
                val_P[i] /= sum_P;
            }
        }

        end = clock();
        clock_gettime(CLOCK_MONOTONIC, &end_p);

        tt = end - start;
        init_real_time += (double)(end_p.tv_sec - start_p.tv_sec) + (double)(end_p.tv_nsec - start_p.tv_nsec)/BILLION;
        init_cpu_time += (double)tt / CLOCKS_PER_SEC;

        printf("PixelSNE: P normalization(P training for VPTree) real time: %.2lf seconds!\n", init_real_time);
        printf("PixelSNE: P normalization(P training for VPTree) clock time: %.2lf seconds!\n", init_cpu_time);
    }

    // Lie about the P-values
    if(exact) { for(int i = 0; i < N * N; i++)        P[i] *= 12.0; }
    else {      for(int i = 0; i < row_P[N]; i++) val_P[i] *= 12.0; }
    
    if (skip_random_init != true) {
        for(int i = 0; i < N * no_dims; i++) {
            Y[i] = rand() % bins;
        }
    }

	// Perform main training loop
    if(exact) printf("PixelSNE: Learning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
    else printf("PixelSNE: Input similarities sparsity = %lf!\nPixelSNE: Learning embedding...\n", (double) row_P[N] / ((double) N * (double) N));

    //init for updatePoints;
    beta = bins * bins * 1e3;
    tree = NULL;
}
void *updateGradientThread(void *_id)
{
	long long id = (long long)_id;
	long long lo = id * *global2_N / num_threads;
	long long hi = (id + 1) * *global2_N / num_threads;
//	printf("%lld %lld %lld\n", id, lo, hi);
	for (long long i = lo * *global2_no_dims; i < hi * *global2_no_dims; ++i)
	{
		//update gains
		global2_gains[i] = (sign(global2_dY[i]) != sign(global2_uY[i])) ? (global2_gains[i] + .2) : (global2_gains[i] * .8);
		if (global2_gains[i] < .01) global2_gains[i] = .01;
		//update gradient
		global2_uY[i] = *global2_momentum * global2_uY[i] - *global2_eta * global2_gains[i] * global2_dY[i];
		global2_Y[i] = global2_Y[i] + global2_uY[i];
	}

	return NULL;
}
int PixelSNE::updatePoints(double* Y, int &N, int no_dims, double &theta, unsigned int &bins, bool threading, bool sleepingg, int iter, int &stop_lying_iter, int &mom_switch_iter, int &max_iter) {
    isSleeping = sleepingg;
    if(sleepingg && skip == NULL)
    {
        skip = (int *) malloc(N * sizeof(int));
        for(int i = 0; i < N; i++)    skip[i]=1;
    }
    
    if(iter == 0)
    {
        start = clock();
        clock_gettime(CLOCK_MONOTONIC, &start_p);   
    }

    if(KNNupdated)
    {
        free(row_P);
        free(col_P);
        free(val_P);
        row_P = new_row_P;
        col_P = new_col_P;
        val_P = new_val_P;
/*
        if(isLogging)
        {
            char buffer[50];
            sprintf(buffer, "P_iter_%d.log", iter+1);
            save_P(buffer);
        }*/ //P is not needed
        if(iter <= stop_lying_iter) {
            if(exact) {  if(P != NULL) for(int i = 0; i < tempN * tempN; i++)        P[i] *= 12.0; }
            else {      for(int i = 0; i < row_P[tempN]; i++) val_P[i] *= 12.0; }
        }
        KNNupdated = false;
        printf("PixelSNE: KNN Updated when iter is %d!\n", iter+1);
    }
    if(threading)
    {
        computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta, beta, bins, iter, n_threads);
        
        global2_N = &N;
        global2_no_dims = &no_dims;
        global2_gains = gains;
        global2_dY = dY;
        global2_uY = uY;
        global2_momentum = &momentum;
        global2_eta = &eta;
        global2_Y = Y;
        
        boost::thread *pt = new boost::thread[num_threads];
		for (long long i = 0; i < num_threads; ++i) pt[i] = boost::thread(updateGradientThread, (void*)i);
		for (long long i = 0; i < num_threads; ++i) pt[i].join();
		delete[] pt;

    }
    else
    {
        if(exact) computeExactGradient(P, Y, N, no_dims, dY);
        else computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta, beta, bins, iter);

        // Update gains
        for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    }
    // Perform gradient update (with momentum and gains)
    if(sleepingg)
    {
        for(int i = 0; i < N * no_dims; i++) {
            if(((skip[i/no_dims])&(-skip[i/no_dims]))==skip[i/no_dims])
                uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            if((i%no_dims)==0&&iter>stop_lying_iter_num+150){
                if(((skip[i/no_dims])&(-skip[i/no_dims]))==skip[i/no_dims]){//have to be checked
                    if(-0.01<=uY[i]&&uY[i]<=0.01) {
                        if(((skip[i/no_dims])&(-skip[i/no_dims]))==skip[i/no_dims]){//to see skip[i] is 2^n
                            skip[i/no_dims]*=4;
                            skip[i/no_dims]--;
                        }
                    }
                    else {
                        skip[i/no_dims]=1;
                    }
                }
            }
        }
        for(int i = 0; i < N * no_dims; i++) 
        {
            if(((skip[i/no_dims])&(-skip[i/no_dims]))==skip[i/no_dims]||skip[i/no_dims]==0)
                Y[i] = Y[i] + uY[i];
        }
    }
    else
    {
        for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;
                // Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
        for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];
    
    }


    beta = minmax(Y, N, no_dims, beta, bins, iter);

    // Stop lying about the P-values after a while, and switch momentum
    if(iter == stop_lying_iter) {
        if(exact) { for(int i = 0; i < N * N; i++)        P[i] /= 12.0; }
        else      { for(int i = 0; i < row_P[N]; i++) val_P[i] /= 12.0; }
    }
    if(iter == mom_switch_iter) momentum = final_momentum;


    end = clock();
    clock_gettime(CLOCK_MONOTONIC, &end_p);

    temptime1 += (double)(end_p.tv_sec - start_p.tv_sec) + (double)(end_p.tv_nsec - start_p.tv_nsec)/BILLION;
    tt = end - start;
    temptime2 += (double)tt / CLOCKS_PER_SEC;
    fitting_real_time += (double)(end_p.tv_sec - start_p.tv_sec) + (double)(end_p.tv_nsec - start_p.tv_nsec)/BILLION;
    fitting_cpu_time += (double)tt / CLOCKS_PER_SEC;

    // Print out progress
    if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {

        double C = .0;
        if(exact) C = evaluateError(P, Y, N, no_dims);
        else      C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta, beta, bins, iter);  // doing approximate computation here!


        printf("PixelSNE: Iteration %d: error is %f (50 iterations in %4.2lf real seconds, %4.2lf clock seconds)\n", iter, C, temptime1, temptime2);

        temptime1 = 0;
        temptime2 = 0;
    }
    if (iter == max_iter -1) {
        // Clean up memory
        free(dY); dY = NULL;
        free(uY); uY = NULL;
        free(gains); gains = NULL;
        free(pos_f); pos_f = NULL;
        free(neg_f); neg_f = NULL;

        if(exact) {free(P); P = NULL;}
        else {
            free(row_P); row_P = NULL;
            free(col_P); col_P = NULL;
            free(val_P); val_P = NULL;
        }
        
        printf("PixelSNE: Initialization: %.2lf real seconds\n", init_real_time);
        printf("PixelSNE: Initialization: %.2lf clock seconds\n", init_cpu_time);
        printf("PixelSNE: Initialization with propagation: %.2lf real seconds\n", init_real_time+propagation_real_time);
        printf("PixelSNE: Initialization with propagation: %.2lf clock seconds\n", init_cpu_time+propagation_cpu_time);
        printf("PixelSNE: Fitting performed in %4.2f real seconds.\n", fitting_real_time);
        printf("PixelSNE: Fitting performed in %4.2f clock seconds!\n", fitting_cpu_time);
        printf("PixelSNE: Total real time: %.2lfs\n", init_real_time + fitting_real_time);
        printf("PixelSNE: Total clock time: %.2lfs\n", init_cpu_time + fitting_cpu_time);
        printf("PixelSNE: Total with propagation real time: %.2lfs\n", init_real_time+propagation_real_time + fitting_real_time);
        printf("PixelSNE: Total with propagation clock time: %.2lfs\n", init_cpu_time+propagation_cpu_time + fitting_cpu_time);
    }
    start = clock();
    clock_gettime(CLOCK_MONOTONIC, &start_p);
    return iter+1;
    //this value is ignored.
}

void *computeNonEdgeForcesThread(void *_id)
{
	long long id = (long long)_id;
	long long lo = id * *global2_N / num_threads;
	long long hi = (id + 1) * *global2_N / num_threads;

	int i;
	global2_sumq[id] = 0;
    global2_skip_cnt[id] = 0;
	for (i = lo; i < hi; ++i)
	{
        if(globalIsSleeping)
        {
            if(global2_skip[i]>1 ) global2_skip[i]--;
            if(((global2_skip[i])&(-global2_skip[i]))==global2_skip[i]||global2_skip[i]==0){
        		gradientTree->computeNonEdgeForces(i, *global2_theta, global2_negf+ i * *global2_D, &global2_sumq[id], *global2_beta, *global2_iter_cnt);
            }
            else{
                global2_skip_cnt[id]++;
            }

        }
        else
        {
    		gradientTree->computeNonEdgeForces(i, *global2_theta, global2_negf+ i * *global2_D, &global2_sumq[id], *global2_beta, *global2_iter_cnt);
        }
	}
	return NULL;
}

//multithread;
void PixelSNE::computeGradient(unsigned long long* inp_row_P, unsigned long long* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, 
    double theta, double beta, unsigned int bins, int iter_cnt, int nthreads){ 

    // Construct space-partitioning tree on current map
    if (tree == NULL){
        tree = new PTree(D, Y, N, bins, 0, iter_cnt);
        //tree->print();
        //printf("PixelSNE: num_insert: %lld\n", num_insert);
    }
    else {
        tree->clean(iter_cnt);
        tree->fill(N, iter_cnt);
    }

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;

    if(pos_f == NULL && neg_f == NULL)
    {
        pos_f = (double*) calloc(N * D, sizeof(double));
        neg_f = (double*) calloc(N * D, sizeof(double));
        if(pos_f == NULL || neg_f == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    }
    else
    {
        for(int i = 0; i < N*D; i++) neg_f[i]=0;
        for(int i = 0; i < N*D; i++) pos_f[i]=0;
    }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f, beta, nthreads);

    global2_negf = neg_f;
    global2_theta = &theta;
    global2_beta = &beta;
    global2_iter_cnt = &iter_cnt;
    global2_N = &N;
    global2_D = &D;
    global2_sumq = (double *) malloc(num_threads * sizeof(double));
    global2_skip_cnt = (int *) malloc(num_threads * sizeof(int));
    global2_skip = skip;
    globalIsSleeping = isSleeping;
    gradientTree = tree;

	boost::thread *pt = new boost::thread[num_threads];
	for (long long i = 0; i < num_threads; ++i) pt[i] = boost::thread(computeNonEdgeForcesThread, (void*)i);
	for (long long i = 0; i < num_threads; ++i) pt[i].join();
	delete[] pt;
	for (long long i = 0; i < num_threads; ++i)
		sum_Q += global2_sumq[i];

    if(isSleeping)
    {
        int cntt = 0;
        for(int i = 0; i < num_threads; ++i) cntt += global2_skip_cnt[i];
        if(cntt != 0) printf("PixelSNE: SGD Skipped : %d\n",cntt);

        // Compute final t-SNE gradient
        for(int i = 0; i < N * D; i++) {
            if(((skip[i/D])&(-skip[i/D]))==skip[i/D]||skip[i/D]==0)
                dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
            else dC[i]=0;
        }
    }
    else
    {
        // Compute final t-SNE gradient 
        for(int i = 0; i < N * D; i++) { 
            dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
        }
    }
    free(global2_sumq); global2_sumq = NULL;
    free(global2_skip_cnt); global2_skip_cnt = NULL;
}

void PixelSNE::computeGradient(unsigned long long* inp_row_P, unsigned long long* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, 
    double theta, double beta, unsigned int bins, int iter_cnt){ 

    // Construct space-partitioning tree on current map
    if (tree == NULL){
        tree = new PTree(D, Y, N, bins, 0, iter_cnt);
        //tree->print();
        //printf("PixelSNE: num_insert: %lld\n", num_insert);
    }
    else {
        tree->clean(iter_cnt);
        tree->fill(N, iter_cnt);
    }

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;

    if(pos_f == NULL && neg_f == NULL)
    {
        pos_f = (double*) calloc(N * D, sizeof(double));
        neg_f = (double*) calloc(N * D, sizeof(double));
        if(pos_f == NULL || neg_f == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    }
    else
    {
        for(int i = 0; i < N*D; i++) neg_f[i]=0;
        for(int i = 0; i < N*D; i++) pos_f[i]=0;
    }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f, beta);
    
    if(isSleeping)
    {
        int cntt=0;
        for(int n = 0; n < N; n++) {
            if(skip[n]>1 ) skip[n]--;
            if(((skip[n])&(-skip[n]))==skip[n]||skip[n]==0){
                tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q, beta, iter_cnt);
            }
            else{
                cntt++;
            }
        }
        if(cntt != 0) printf("PixelSNE: SGD Skipped : %d\n",cntt);
        // Compute final t-SNE gradient
        for(int i = 0; i < N * D; i++) {
            if(((skip[i/D])&(-skip[i/D]))==skip[i/D]||skip[i/D]==0)
                dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
            else dC[i]=0;
        }
    }
    else
    {
        for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q, beta, iter_cnt); 
        // Compute final t-SNE gradient 
        for(int i = 0; i < N * D; i++) { 
            dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
        }
    }
}

// Compute gradient of the t-SNE cost function (exact)
void PixelSNE::computeExactGradient(double* P, double* Y, int N, int D, double* dC) {

	// Make sure the current gradient contains zeros
	for(int i = 0; i < N * D; i++) dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    double* Q    = (double*) malloc(N * N * sizeof(double));
    if(Q == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    double sum_Q = .0;
    int nN = 0;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

	// Perform the computation of the gradient
    nN = 0;
    int nD = 0;
	for(int n = 0; n < N; n++) {
        int mD = 0;
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
                for(int d = 0; d < D; d++) {
                    dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
                }
            }
            mD += D;
		}
        nN += N;
        nD += D;
	}

    // Free memory
    free(DD); DD = NULL;
    free(Q);  Q  = NULL;
}


// Evaluate t-SNE cost function (exactly)
double PixelSNE::evaluateError(double* P, double* Y, int N, int D) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    double* Q = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL || Q == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    int nN = 0;
    double sum_Q = DBL_MIN;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
            else Q[nN + m] = DBL_MIN;
        }
        nN += N;
    }
    for(int i = 0; i < N * N; i++) Q[i] /= sum_Q;

    // Sum t-SNE error
    double C = .0;
	for(int n = 0; n < N * N; n++) {
        C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
	}

    // Clean up memory
    free(DD);
    free(Q);
	return C;
}

void *computeNonEdgeForcesForErrorThread(void *_id)
{
	long long id = (long long)_id;
	long long lo = id * *global2_N / num_threads;
	long long hi = (id + 1) * *global2_N / num_threads;

	int i;
//	double sum_Q;
	global2_sumq[id] = 0;
	for (i = lo; i < hi; ++i)
	{
		ptree->computeNonEdgeForces(i, *global2_theta, global2_negf, &global2_sumq[id], *global2_beta, *global2_iter_cnt);
//		global2_sumq[id] += sum_Q;
	}

	return NULL;
}

// Evaluate t-SNE cost function (approximately)
double PixelSNE::evaluateError(unsigned long long* row_P, unsigned long long* col_P, double* val_P, double* Y, int N, int D, double theta, double beta, unsigned int bins, int iter_cnt)
{

    // Get estimate of normalization term
    ptree = new PTree(D, Y, N, bins, 0, iter_cnt);
    global2_negf = (double*) calloc(D, sizeof(double));
    global2_theta = &theta;
    global2_beta = &beta;
    global2_iter_cnt = &iter_cnt;
    global2_N = &N;
    global2_sumq = (double *) malloc(num_threads * sizeof(double));

    double sum_Q = .0;

	boost::thread *pt = new boost::thread[num_threads];
	for (long long i = 0; i < num_threads; ++i) pt[i] = boost::thread(computeNonEdgeForcesForErrorThread, (void*)i);
	for (long long i = 0; i < num_threads; ++i) pt[i].join();
	delete[] pt;
	for (long long i = 0; i < num_threads; ++i)
		sum_Q += global2_sumq[i];

    // Loop over all edges to compute t-SNE error
    int ind1, ind2;
    double C = .0, Q;
    for(int n = 0; n < N; n++) {
        ind1 = n * D;
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * D;
            for(int d = 0; d < D; d++) global2_negf[d]  = Y[ind1 + d];
            for(int d = 0; d < D; d++) global2_negf[d] -= Y[ind2 + d];
            for(int d = 0; d < D; d++) Q += global2_negf[d] * global2_negf[d];
            Q = (beta / (beta + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));

        }
    }

    // Clean up memory
    free(global2_negf);
    free(global2_sumq);
    delete ptree;
    return C;
}


// Compute input similarities with a fixed perplexity
void PixelSNE::computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity) {    

	// Compute the squared Euclidean distance matrixgoog
	double* DD = (double*) malloc(N * N * sizeof(double)); // symmetric metrix
    if(DD == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(X, N, D, DD);          // time complexity: O(N^2)

	// Compute the Gaussian kernel row by row
    int nN = 0;
	for(int n = 0; n < N; n++) {

		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;
        double sum_P;

		// Iterate until we found a good perplexity
		int iter = 0;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
            #ifdef USE_BITWISE_OP
    			for (int m = 0; m < N; m++) P[nN + m] = fexp(-beta * DD[nN + m]);/*Op*/
            #else
                for(int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);/*Op*/
            #endif
			P[nN + n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += P[nN + m];
			double H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row normalize P
		for(int m = 0; m < N; m++) P[nN + m] /= sum_P;
        nN += N;
	}

	// Clean up memory
	free(DD); DD = NULL;
}


// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void PixelSNE::computeGaussianPerplexity(double* X, int N, int D, unsigned long long** _row_P, unsigned long long** _col_P, double** _val_P, double perplexity, int K) {

    if(perplexity > K) printf("PixelSNE: Perplexity should be lower than K!\n");


    // Allocate the memory we need
    *_row_P = (unsigned long long*)    malloc((N + 1) * sizeof(unsigned long long));
    *_col_P = (unsigned long long*)    calloc(N * K, sizeof(unsigned long long));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    unsigned long long* row_P = *_row_P;
    unsigned long long* col_P = *_col_P;
    double* val_P = *_val_P;
    double* cur_P = (double*) malloc((N - 1) * sizeof(double));
    if(cur_P == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned long long) K;

    // Build ball tree on data set(Vantage-point tree) --> time complexity: O(uNlogN)
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    printf("PixelSNE: Building tree...\n");
    vector<DataPoint> indices;
    vector<double> distances;
    for(int n = 0; n < N; n++) {

        if(n % 10000 == 0) printf("PixelSNE:  - point %d of %d\n", n, N);

        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while(!found && iter < 200) {

            // Compute Gaussian kernel row
            #ifdef USE_BITWISE_OP
    			for (int m = 0; m < K; m++) cur_P[m] = fexp(-beta * distances[m + 1] * distances[m + 1]);/*Op*/
            #else
                for (int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);/*Op*/
            #endif

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for(int m = 0; m < K; m++) sum_P += cur_P[m];
            double H = .0;
            for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(unsigned int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }

    // Clean up memory
    obj_X.clear();
    free(cur_P);
    delete tree;
}


// Symmetrizes a sparse matrix
void PixelSNE::symmetrizeMatrix(unsigned long long** _row_P, unsigned long long** _col_P, double** _val_P, int N) {

    // Get sparse matrix
    unsigned long long* row_P = *_row_P;
    unsigned long long* col_P = *_col_P;
    double* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) present = true;
            }
            if(present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for(int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    unsigned long long* sym_row_P = (unsigned long long*) malloc((N + 1) * sizeof(unsigned long long));
    unsigned long long* sym_col_P = (unsigned long long*) malloc(no_elem * sizeof(unsigned long long));
    double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
    if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                        

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) {
                    present = true;
                    if(n <= col_P[i]) {                                                 
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if(!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if(!present || (present && n <= col_P[i])) {
                offset[n]++;
                if(col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix
void PixelSNE::computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
    const double* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const double* XmD = XnD + D;
        double* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;                       
        double* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);    
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}


// Makes data zero-mean
void PixelSNE::zeroMean(double* X, int N, int D) {
	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
    if(mean == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}

	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}
    free(mean); mean = NULL;
}

double PixelSNE::minmax(double* X, int N, int D, double beta, unsigned int bins, int iter_cnt) {
    
    // Compute data min, max
    double ran, xran, yran; 
    double* min = (double*) calloc(D, sizeof(double));
    double* max = (double*) calloc(D, sizeof(double));
    if(min == NULL || max == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }

    for (int d = 0 ; d < D; d++) {
        min[d] = INT_MAX;
        max[d] = INT_MIN;
    }

    int nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            if (min[d] > X[nD + d]) {
                min[d] = X[nD + d];
            }
        }
        nD += D;
    }

    nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            X[nD + d] -= min[d];
        }
        nD += D;
    }

    nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            if (max[d] < X[nD + d]) {
                max[d] = X[nD + d];
            }
        }
        nD += D;
    }

    xran = float(bins-1) / (max[0]);
    yran = float(bins-1) / (max[1]);

    ran = (xran < yran) ? xran : yran;
    beta = beta * pow(ran, 2);

    // Subtract min, max
    nD = 0;
    for(int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            
            if (0 == d) {
                X[nD + d] = X[nD + d] * xran ;
            }
            else{
                X[nD + d] = X[nD + d] * yran ;
            }
        }
        nD += D;
    }
    nD = 0;

    free(min); free(max); min = NULL; max = NULL;
    return beta;
}


// Generates a Gaussian random number
double PixelSNE::randn() {
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

void PixelSNE::load_data(const char* inputfile, double **data, int* n, int* d)
{
    int res = -1;
	FILE *fin = fopen(inputfile, "rb");
	if (fin == NULL)
	{
		printf("PixelSNE: File not found!\n");
		return;
	}
    printf("PixelSNE: Reading input file %s ......\n", inputfile);
    res = fscanf(fin, "%d%d", n, d);
//    dddd = new int(asdfasdf);    
//	res = fscanf(fin, "%d%d", nnn, dddd);
    double *tmpData = new double[*n * *d];
	for (int i = 0; i < *n; ++i)
	{
		for (int j = 0; j < *d; ++j)
		{
			res = fscanf(fin, "%lf", &tmpData[i * *d + j]);
		}
	}
    *data = tmpData;
	fclose(fin);
	printf("PixelSNE:  Done.\n");
	printf("PixelSNE: Total vertices : %d\tDimension : %d\n", *n, *d);
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool PixelSNE::load_data(const char* inputfile, double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, unsigned int* bins, int* p_method, int* rand_seed) {

    // Open file, read first 2 integers, allocate memory, and read the data
    FILE *h;
    size_t res = -1;
    if((h = fopen(inputfile, "r+b")) == NULL) {
        printf("PixelSNE: Error: could not open data file.\n");
        return false;
    }
    res = fread(n, sizeof(int), 1, h);                                            
    res = fread(d, sizeof(int), 1, h);                                            
    res = fread(theta, sizeof(double), 1, h);                                    
    res = fread(perplexity, sizeof(double), 1, h);                               
    res = fread(no_dims, sizeof(int), 1, h);                                      
    res = fread(p_method, sizeof(int), 1, h);                                     
    res = fread(bins, sizeof(unsigned int), 1, h);

    *data = (double*) malloc(*d * *n * sizeof(double));
    if(*data == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
    res = fread(*data, sizeof(double), *n * *d, h);                               

    if(!feof(h)) res = fread(rand_seed, sizeof(int), 1, h);                      
    fclose(h);
    printf("PixelSNE: Read the %i x %i data matrix successfully!\n", *n, *d);


    return true;
}


// Function that saves map to a t-SNE file
void PixelSNE::save_data(double* data, int* landmarks, double* costs, int n, int d) {

	// Open file, write first 2 integers and then the data
	FILE *h;
	if((h = fopen("result.dat", "w+b")) == NULL) {
		printf("PixelSNE: Error: could not open data file.\n");
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
    fwrite(data, sizeof(double), n * d, h);
	fwrite(landmarks, sizeof(int), n, h);
    fwrite(costs, sizeof(double), n, h);
    fclose(h);
	printf("PixelSNE: Wrote the %i x %i data matrix successfully!\n", n, d);
}

void PixelSNE::save_data(const char* outfile, double* Y, int N, int D, double theta, unsigned int bins, int iter)
{
    char tempname[1000];
    char buffer[50];
    sprintf(buffer, "_iter_%d.log", iter+1);
    strcpy(tempname, outfile);
    strcat(tempname, buffer);
	FILE *h;
	if((h = fopen(tempname, "w+b")) == NULL) {
		printf("PixelSNE: Error: could not open data file.\n");
		return;
	}
	fwrite(&N, sizeof(int), 1, h);
	fwrite(&D, sizeof(int), 1, h);
    fwrite(Y, sizeof(double), N * D, h);
    double C = 0;
    C = evaluateError(row_P, col_P, val_P, Y, N, D, theta, beta, bins, iter);
    fwrite(&C, sizeof(double), 1, h);

    fclose(h);
/*    if(!isLogging)
    {
        strcpy(tempname, "P_original.log");
        save_P(tempname);    
    }*/ //P is not needed
    isLogging = true;

	printf("PixelSNE: Wrote the %i x %i data matrix successfully!\n", N, D);    
}

void PixelSNE::updateKNN(int i)
{
    p_model->run_propagation_once(i, knn_validation); 
	p_model->get_result(&new_row_P, &new_col_P, &new_val_P);

    start2 = clock();
    clock_gettime(CLOCK_MONOTONIC, &start_p2);

	double sum_P = .0;
	for(int i = 0; i < new_row_P[tempN]; i++) {
		sum_P += new_val_P[i];
	}
	for(int i = 0; i < new_row_P[tempN]; i++) {
		new_val_P[i] /= sum_P;
	}

	end2 = clock();
	clock_gettime(CLOCK_MONOTONIC, &end_p2);

	tt2 = end2 - start2;
    double temptime11 = (double)(end_p2.tv_sec - start_p2.tv_sec) + (double)(end_p2.tv_nsec - start_p2.tv_nsec)/BILLION;    
    double temptime22 = (double)tt2 / CLOCKS_PER_SEC;

	printf("PixelSNE: P normalization of Updated KNN %d: %.2lf real seconds!\n", i + 1, temptime11);
	printf("PixelSNE: P normalization of Updated KNN %d: %.2lf clock seconds!\n", i + 1, temptime22);

    double* largeVisRealTime = p_model->get_real_time();
    double* largeVisClockTime = p_model->get_clock_time();
    propagation_real_time += (largeVisRealTime[i*2+2] + largeVisRealTime[i*2+3]);
    propagation_cpu_time += (largeVisClockTime[i*2+2] + largeVisClockTime[i*2+3]);

    propagation_cpu_time += temptime22;
    propagation_real_time += temptime11;

    KNNupdated = true;
}

int PixelSNE::get_propagation_num()
{
	return (int)n_propagations;
}

int PixelSNE::get_max_iter()
{
	return (int)max_iteration;
}

void PixelSNE::save_P(char *filename)
{
    FILE* fp_saved = fopen(filename, "w+");
    char temp_str[100] = "";

    int idx = 0;
    for(int n = 0; n < tempN; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {
            

            sprintf(temp_str, "%lld %lld %f\n", row_P[n], col_P[idx], val_P[idx]);
            fwrite(temp_str, strlen(temp_str), 1, fp_saved);

            ++idx;
        }
    }

    fclose(fp_saved);
}


/*
// Function that runs the Barnes-Hut implementation of t-SNE
int main() {

    // Define some variables
    int     origN;                  
    int     N;                      
    int     D;                      
    int     no_dims = 2;                
    int*    landmarks;              
	double  perc_landmarks;         
    double  perplexity = 30;
    double  theta = 0.5;                
    double* data;                   
    unsigned int bins;
    int     p_method;
    int rand_seed = 30;            
    PixelSNE* pixelsne = new PixelSNE();

    // #ifdef USE_BITWISE_OP
    //     printf("PixelSNE: pixelsne.cpp USE_BITWISE_OP\n");
    // #else
    //     printf("PixelSNE: pixelsne.cpp not USE_BITWISE_OP\n");
    // #endif

	//Op
	pexp = (double*)calloc(EXP_LUT_DIV, sizeof(double));
	for (int i = 0; i < EXP_LUT_DIV; i++)
	{
        #ifdef USE_BITWISE_OP
            pexp[i] = exp((double)EXP_LUT_RANGE * ((double)(i << 1) / EXP_LUT_DIV - 1));
        #else
            pexp[i] = exp((double)EXP_LUT_RANGE * ((double)(i * 2) / EXP_LUT_DIV - 1));
        #endif
	}

    // Read the parameters and the dataset
	if(pixelsne->load_data(&data, &origN, &D, &no_dims, &theta, &perplexity, &bins, &p_method, &rand_seed)) {
		// Make dummy landmarks
        N = origN;

        int* landmarks = (int*) malloc(N * sizeof(int));        
        if(landmarks == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
        for(int n = 0; n < N; n++) landmarks[n] = n;            

        double* Y = (double*) malloc(N * no_dims * sizeof(double)); 
		double* costs = (double*) calloc(N, sizeof(double));         
        if(Y == NULL || costs == NULL) { printf("PixelSNE: Memory allocation failed!\n"); exit(1); }
        
        pixelsne->run(data, N, D, Y, no_dims, perplexity, theta, bins, p_method, rand_seed, false);
		pixelsne->save_data(Y, landmarks, costs, N, no_dims);

        // Clean up the memory
        if (data != NULL){
            free(data); data = NULL;    
        }
		free(Y); Y = NULL;
		free(costs); costs = NULL;
		free(landmarks); landmarks = NULL;
    }
    delete(pixelsne);
}
*/