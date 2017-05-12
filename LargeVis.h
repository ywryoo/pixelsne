/*
   Copyright 2016 LargeVis authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Modified by Ryangwook Ryoo
*/

#ifndef LARGEVIS_H
#define LARGEVIS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "ANNOY/annoylib.h"
#include "ANNOY/kissrandom.h"

#include <pthread.h>
#include <gsl/gsl_rng.h>

typedef float real;

struct arg_struct{
	void *ptr;
	int id;
	arg_struct(void *x, int y) :ptr(x), id(y){}
};

class LargeVis{
private:
	void* self;
	long long n_vertices, n_dim, out_dim, n_samples, n_threads, n_negatives, n_neighbors, n_trees, n_propagations, edge_count_actual;
	real initial_alpha, gamma, perplexity;
	real *vec, *vis;
	std::vector<string> names;
	std::vector<int> *knn_vec, *old_knn_vec;
	AnnoyIndex<int, real, Euclidean, Kiss64Random> *annoy_index;
	long long n_edge, *head;
    std::vector<long long> next, reverse;
    std::vector<int> edge_from, edge_to;
	std::vector<real> edge_weight;
    int *neg_table;
    long long neg_size;
	long long *alias;
	real *prob;
	static const gsl_rng_type * gsl_T;
	static gsl_rng * gsl_r;
	bool *knn_not_changed;
	double real_time[50] = {0};
	double cpu_time[50] = {0};
	//0: normalize + rptrees(annoy), 1: knn->P for 0
	//0 + 1: init time
	//2: propagation 1, 3: knn->P for 2
	//2 + 3: propagation 1 + calculate P time
	//4: 5: ... so on
	struct timespec sstart_p, eend_p;
	clock_t ttt;

	void clean_model();
	void clean_data();
	void clean_graph();
	void normalize();
	real CalcDist(long long x, long long y);
	void run_annoy();
	void annoy_thread(int id);
	static void *annoy_thread_caller(void *arg);
	void run_propagation();
	void propagation_thread(int id);
	static void *propagation_thread_caller(void *arg);
	void test_accuracy();
	void compute_similarity();
	void compute_similarity_thread(int id);
	static void *compute_similarity_thread_caller(void *arg);
	void search_reverse_thread(int id);
	static void *search_reverse_thread_caller(void *arg);
	void construt_knn();

public:
	LargeVis();
	void load_from_file(char *infile);
	void load_from_graph(char *infile);
	void load_from_data(real *data, long long n_vert, long long n_di);
	void load_from_data(double *data, long long n_vert, long long n_di);
	void save(char *outfile);
	void run(long long out_d = -1, long long n_thre = -1, long long n_samp = -1, long long n_prop = -1, real alph = -1, long long n_tree = -1, long long n_nega = -1, long long n_neig = -1, real gamm = -1, real perp = -1);
	real *get_ans();
	long long get_n_vertices();
	long long get_out_dim();
	void get_result(unsigned long long** row_P, unsigned long long** col_P, double** val_P);
	void run_propagation_once(int i, bool knn_validation);
	double* get_real_time();
	double* get_clock_time();
	void setThreadsNum(long long i);
};

#endif