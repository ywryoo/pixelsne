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
	struct timespec start_p2, end_p2;
    struct timespec start_p3, end_p3;
    bool exact;
    PTree* tree;
    float total_time;
    double total_time2;
    double total_time3;
    clock_t start, end;
    double momentum, final_momentum;
    double eta;
    double* dY;
    double* uY;
    double* gains;
    double* P;
    double* pos_f = NULL;
    double* neg_f = NULL;
    int tempN;
    unsigned long long* row_P;        
    unsigned long long* col_P;
    double* val_P;
    unsigned long long* new_row_P;        
    unsigned long long* new_col_P;
    double* new_val_P;
    
    double beta;
    LargeVis* p_model;
    bool KNNupdated;
    bool knn_validation;
    int max_iteration;
    int n_propagations;
public:
    PixelSNE();
    ~PixelSNE();
    void run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta,
        unsigned int bins, int p_method, int rand_seed, int n_threads, int propagation_num, bool skip_random_init, int n_trees, bool isValidation, int max_iter=1000, int stop_lying_iter=250, 
        int mom_switch_iter=250);
    void load_data(const char* inputfile, double **data, int* n, int* d);
    bool load_data(const char* inputfile, double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, unsigned int* bins, int* p_method, int* rand_seed);
    void save_data(double* data, int* landmarks, double* costs, int n, int d);
    void symmetrizeMatrix(unsigned long long** row_P, unsigned long long** col_P, double** val_P, int N); // should be static!
    int updatePoints(double* Y, int &N, int no_dims, double &theta, unsigned int &bins, int iter, int &stop_lying_iter, int &mom_switch_iter, int &max_iter);
    void updateKNN(int i);
    int get_propagation_num();
    int get_max_iter();
private:
    void computeGradient(double* P, unsigned long long* inp_row_P, unsigned long long* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, double beta, unsigned int bins, int iter_cnt);
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
