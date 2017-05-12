/// from PixelSNE: https://github.com/awesome-davian/pixelsne
/// Modified by Ryangwook Ryoo, 2017

#ifndef PIXELSNE_H
#define PIXELSNE_H

#include "ptree.h"
#include "LargeVis.h"
#include <stdlib.h>
#include <stdio.h>

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

class PixelSNE
{
private:
	struct timespec start_p, end_p;
    struct timespec start_p2, end_p2;
    bool exact;
    bool isLogging;
    bool isSleeping;
    bool propDone;
    bool isPipelined;
    PTree* tree;
    clock_t start, end;
    clock_t start2, end2;
    clock_t tt;
    clock_t tt2;
    double momentum, final_momentum;
    double eta;
    double* dY;
    double* uY;
    double* gains;
    double* P;
    double temptime1;
    double temptime2;
    double* pos_f = NULL;
    double* neg_f = NULL;
    int tempN;
    unsigned long long* row_P;        
    unsigned long long* col_P;
    double* val_P;
    unsigned long long* new_row_P;        
    unsigned long long* new_col_P;
    double* new_val_P;
    int originalThreads;
    double beta;
    LargeVis* p_model;
    bool KNNupdated;
    bool knn_validation;
    int max_iteration;
    int n_propagations;
    int n_threads;
    int *skip;
    int stop_lying_iter_num;
public:
    double fitting_cpu_time;
    double fitting_real_time;
    double propagation_cpu_time;
    double propagation_real_time;
    double init_real_time;
    double init_cpu_time;
    PixelSNE();
    ~PixelSNE();
    void run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta,
        unsigned int bins, int p_method, int rand_seed, int n_threads, int propagation_num, bool skip_random_init, int n_trees, bool isValidation, bool pipelined ,int max_iter=1000, int stop_lying_iter=250, 
        int mom_switch_iter=250);
    void load_data(const char* inputfile, double **data, int* n, int* d);
    bool load_data(const char* inputfile, double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, unsigned int* bins, int* p_method, int* rand_seed);
    void save_data(double* data, int* landmarks, double* costs, int n, int d);
    void save_data(const char* outfile, double* Y, int N, int D, double theta, unsigned int bins, int iter);
    void symmetrizeMatrix(unsigned long long** row_P, unsigned long long** col_P, double** val_P, int N); // should be static!
    int updatePoints(double* Y, int &N, int no_dims, double &theta, unsigned int &bins, bool isthreaded, bool sleeping, int iter, int &stop_lying_iter, int &mom_switch_iter, int &max_iter);
    void updateKNN(int i);
    int get_propagation_num();
    int get_max_iter();
    void save_P(char *filename);
private:
    void computeGradient(unsigned long long* inp_row_P, unsigned long long* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, double beta, unsigned int bins, int iter_cnt);
    void computeGradient(unsigned long long* inp_row_P, unsigned long long* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, double beta, unsigned int bins, int iter_cnt, int nthreads);
    void computeExactGradient(double* P, double* Y, int N, int D, double* dC);
    double evaluateError(double* P, double* Y, int N, int D);
    double evaluateError(unsigned long long* row_P, unsigned long long* col_P, double* val_P, double* Y, int N, int D, double theta, double beta, unsigned int bins, int iter_cnt);
    void zeroMean(double* X, int N, int D);
    double minmax(double* X, int N, int D, double beta, unsigned int bins, int iter_cnt);
    void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity);
    void computeGaussianPerplexity(double* X, int N, int D, unsigned long long** _row_P, unsigned long long** _col_P, double** _val_P, double perplexity, int K);
    void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
    double randn();

};

#endif
