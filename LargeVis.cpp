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

#include "LargeVis.h"
#include <map>
#include <float.h>
#define BILLION 1000000000L
#define METHOD 1
using namespace similarity;
/*
 * Define an implementation of the distance function.
 */
struct DistL2 {
  /*
   * Important: the function is const and arguments are const as well!!!
   */
  float operator()(const float* x, const float* y, size_t qty) const {
    float res = 0;
    for (size_t i = 0; i < qty; ++i) res+=(x[i]-y[i])*(x[i]-y[i]);
    return sqrt(res);
  }
};
clock_t sstart, eend;
LargeVis::LargeVis()
{
	vec = vis = prob = NULL;
	knn_vec = old_knn_vec = NULL;
	annoy_index = NULL;
	head = alias = NULL;
    neg_table = NULL;
	knn_not_changed = NULL;
}

const gsl_rng_type *LargeVis::gsl_T = NULL;
gsl_rng *LargeVis::gsl_r = NULL;

void LargeVis::clean_model()
{
	if (vis) delete[] vis;
	if (prob) delete[] prob;
	if (knn_vec) delete[] knn_vec;
	if (old_knn_vec) delete[] old_knn_vec;
	if (annoy_index) delete annoy_index;
	if (neg_table) delete[] neg_table;
	if (alias) delete[] alias;
	if (knn_not_changed) delete[] knn_not_changed; knn_not_changed = NULL;
	vis = prob = NULL;
	knn_vec = old_knn_vec = NULL;
	annoy_index = NULL;
    neg_table = NULL;
    alias = NULL;

	edge_count_actual = 0;
	neg_size = 1e8;
}

void LargeVis::clean_graph()
{
	if (head) { delete[] head; head = NULL; }

	n_edge = 0;
	next.clear(); edge_from.clear(); edge_to.clear(); reverse.clear(); edge_weight.clear(); names.clear();
}

void LargeVis::clean_data()
{
	if (vec) { delete[] vec; vec = NULL; }
	clean_graph();
}

void LargeVis::load_from_file(char *infile)
{
	int res = -1;
	clean_data();
	FILE *fin = fopen(infile, "rb");
	if (fin == NULL)
	{
		printf("LargeVis: File not found!\n");
		return;
	}
    printf("LargeVis: Reading input file %s ......\n", infile);
	res = fscanf(fin, "%lld%lld", &n_vertices, &n_dim);
	vec = new real[n_vertices * n_dim];
	for (long long i = 0; i < n_vertices; ++i)
	{
		for (long long j = 0; j < n_dim; ++j)
		{
			res = fscanf(fin, "%f", &vec[i * n_dim + j]);
		}
	}
	fclose(fin);
	printf("LargeVis: Reading input file Done.\n");
	printf("LargeVis: Total vertices : %lld\tDimensions : %lld\n", n_vertices, n_dim);
}

void LargeVis::load_from_data(real *data, long long n_vert, long long n_di)
{
	clean_data();
	vec = data;
	n_vertices = n_vert;
	n_dim = n_di;
	printf("LargeVis: Total vertices : %lld\tDimensions : %lld\n", n_vertices, n_dim);
}

void LargeVis::load_from_data(double *data, long long n_vert, long long n_di)
{
	clean_data();
	

	vec = new real[n_vert * n_di];
    if(vec == NULL) { 
    	printf("LargeVis: Memory allocation failed!\n"); exit(1);
    }

	//int nD = 0;
	for(long long i = 0; i < n_vert; i++) {
		for(long long j = 0; j < n_di; j++) {
			vec[i * n_di + j] = data[i * n_di + j];

		}
	}
	n_vertices = n_vert;
	n_dim = n_di;
	printf("LargeVis: Total vertices : %lld\tDimension : %lld\n", n_vertices, n_dim);
}

void LargeVis::load_from_graph(char *infile)
{
	clean_data();
	char *w1 = new char[1000];
	char *w2 = new char[10000];
	long long x, y, i, p;
	real weight;
	std::map<std::string, long long> dict;
	n_vertices = 0;
	FILE *fin = fopen(infile, "rb");
	if (fin == NULL)
	{
		printf("LargeVis: File not found!\n");
		return;
	}
	printf("LargeVis: Reading input file %s ......%c", infile, 13);
	while (fscanf(fin, "%s%s%f", w1, w2, &weight) == 3)
	{
		if (!dict.count(w1)) { dict[w1] = n_vertices++; names.push_back(w1); }
		if (!dict.count(w2)) { dict[w2] = n_vertices++; names.push_back(w2); }
		x = dict[w1];
		y = dict[w2];
		edge_from.push_back(x);
		edge_to.push_back(y);
		edge_weight.push_back(weight);
		next.push_back(-1);
		++n_edge;
		if (n_edge % 5000 == 0)
		{
			printf("LargeVis: Reading input file %s ...... %lldK edges%c\n", infile, n_edge / 1000, 13);
		}
	}
	fclose(fin);
	delete[] w1;
	delete[] w2;

	head = new long long[n_vertices];
	for (i = 0; i < n_vertices; ++i) head[i] = -1;
	for (p = 0; p < n_edge; ++p)
	{
		x = edge_from[p];
		next[p] = head[x];
		head[x] = p;
	}
	printf("LargeVis: \nTotal vertices : %lld\tTotal edges : %lld\n", n_vertices, n_edge);
}

void LargeVis::save(char *outfile)
{
	FILE *fout = fopen(outfile, "wb");
	fprintf(fout, "%lld %lld\n", n_vertices, out_dim);
	for (long long i = 0; i < n_vertices; ++i)
	{
		if (names.size()) fprintf(fout, "%s ", names[i].c_str());
		for (long long j = 0; j < out_dim; ++j)
		{
			if (j) fprintf(fout, " ");
			fprintf(fout, "%.6f", vis[i * out_dim + j]);
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}

real *LargeVis::get_ans()
{
	return vis;
}

long long LargeVis::get_n_vertices()
{
	return n_vertices;
}

long long LargeVis::get_out_dim()
{
	return out_dim;
}

void LargeVis::normalize()
{
    printf("LargeVis: Normalizing ......\n");
	real *mean = new real[n_dim];
	for (long long i = 0; i < n_dim; ++i) mean[i] = 0;
	for (long long i = 0, ll = 0; i < n_vertices; ++i, ll += n_dim)
	{
		for (long long j = 0; j < n_dim; ++j)
			mean[j] += vec[ll + j];
	}
	for (long long j = 0; j < n_dim; ++j)
		mean[j] /= n_vertices;
	real mX = 0;
	for (long long i = 0, ll = 0; i < n_vertices; ++i, ll += n_dim)
	{
		for (long long j = 0; j < n_dim; ++j)
		{
			vec[ll + j] -= mean[j];
			if (fabs(vec[ll + j]) > mX)	mX = fabs(vec[ll + j]);
		}
	}
	for (long long i = 0; i < n_vertices * n_dim; ++i)
		vec[i] /= mX;
	delete[] mean;
	printf("LargeVis: Normalizing Done.\n");
}

real LargeVis::CalcDist(long long x, long long y)
{
	real ret = 0;
	long long i, lx = x * n_dim, ly = y * n_dim;
	for (i = 0; i < n_dim; ++i)
		ret += (vec[lx + i] - vec[ly + i]) * (vec[lx + i] - vec[ly + i]);
	return ret;
}

void LargeVis::annoy_thread(int id)
{
	long long lo = id * n_vertices / n_threads;
	long long hi = (id + 1) * n_vertices / n_threads;
	if(METHOD == 0)
	{
		AnnoyIndex<int, real, Euclidean, Kiss64Random> *cur_annoy_index = NULL;
		if (id > 0)
		{
			cur_annoy_index = new AnnoyIndex<int, real, Euclidean, Kiss64Random>(n_dim);
			cur_annoy_index->load("annoy_index_file");
		}
		else
			cur_annoy_index = annoy_index;
		for (long long i = lo; i < hi; ++i)
		{
			cur_annoy_index->get_nns_by_item(i, n_neighbors + 1, (n_neighbors + 1) * n_trees, &knn_vec[i], NULL);
			for (long long j = 0; j < knn_vec[i].size(); ++j)
				if (knn_vec[i][j] == i)
				{
					knn_vec[i].erase(knn_vec[i].begin() + j);
					break;
				}
		}
		if (id > 0) delete cur_annoy_index;
	}
	else
	{
		Index<float>* cur_index;
		unsigned K = 5; // 5-NN query
		cur_index->LoadIndex("HnswIndex");
		VectorSpaceGen<float, DistL2>   customSpace;
		for (long long i = lo; i < hi; ++i)
		{
			
			KNNQuery<float>   knnQ(customSpace, dataSet[i], K);
			cur_index->Search(&knnQ);

		}
		  vector<string>                  vExternIds;
		vExternIds.resize(dataSet.size()); 
		customSpace.WriteDataset(dataSet, vExternIds, "testdataset.txt");

		delete cur_index;cur_index=NULL;
	}


}

void *LargeVis::annoy_thread_caller(void *arg)
{
	LargeVis *ptr = (LargeVis*)(((arg_struct*)arg)->ptr);
	ptr->annoy_thread(((arg_struct*)arg)->id);
	pthread_exit(NULL);
}

void LargeVis::run_annoy()
{
	if(METHOD == 0)
	{
		printf("LargeVis: n_trees: %d\n", n_trees);
		printf("LargeVis: Running ANNOY(Generating RP Trees) ......\n");
		annoy_index = new AnnoyIndex<int, real, Euclidean, Kiss64Random>(n_dim);
		
		
		for (long long i = 0; i < n_vertices; ++i)
			annoy_index->add_item(i, &vec[i * n_dim]);
		annoy_index->build(n_trees);
		if (n_threads > 1) annoy_index->save("annoy_index_file");

		knn_vec = new std::vector<int>[n_vertices];
		
		pthread_t *pt = new pthread_t[n_threads];
		for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::annoy_thread_caller, new arg_struct(this, j));
		for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
		delete[] pt;
		delete annoy_index; annoy_index = NULL;
		printf("LargeVis: Running ANNOY(Generating RP Trees) Done.\n");
	}
	else
	{
		printf("LargeVis: HNSW On\n");
		std::vector<std::vector<float>>  rawData;

		VectorSpaceGen<float, DistL2>   customSpace;
		for(long long i = 0; i < n_vertices; ++i)
		{
			std::vector<float> temp;
			for(long long j = 0; j < n_dim; ++j)
			{
				temp.push_back(vec[i*n_dim+j]);
			}
			rawData.push_back(temp);

		}
		vector<LabelType> labels(rawData.size()); 
		customSpace.CreateDataset(dataSet, rawData, labels); 
		initLibrary(LIB_LOGFILE, "logfile.txt"); 
		AnyParams IndexParams(
								{
								"M=10",
								"indexThreadQty=4" /* 4 indexing threads */
								});

		AnyParams QueryTimeParams( { "efSearch=10", "efSearch=20", "efSearch=40", "efSearch=80", "efSearch=160", "efSearch=240" });

		Index<float>* HnswIndex = 
                        MethodFactoryRegistry<float>::Instance().
                                CreateMethod(false /* don't print progress */,
                                        "hnsw",
                                        "custom",
                                        customSpace,
                                        dataSet);
		HnswIndex->CreateIndex(IndexParams);
		HnswIndex->SetQueryTimeParams(QueryTimeParams);
		HnswIndex->SaveIndex("HnswIndex");
		
		knn_vec = new std::vector<int>[n_vertices];
		
		pthread_t *pt = new pthread_t[n_threads];
		for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::annoy_thread_caller, new arg_struct(this, j));
		for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
		delete[] pt;
		//delete[] rawData; rawData = NULL;
		//delete dataSet; dataSet = NULL;
	}

}

void LargeVis::propagation_thread(int id)
{
	long long lo = id * n_vertices / n_threads;
	long long hi = (id + 1) * n_vertices / n_threads;
	int *check = new int[n_vertices];
	std::priority_queue< pair<real, int> > heap;
	long long x, y, i, j, l1, l2;
	for (x = 0; x < n_vertices; ++x) check[x] = -1;
	for (x = lo; x < hi; ++x)
	{
		if(knn_not_changed[x])
		{
			knn_vec[x] = old_knn_vec[x];
			continue;
		}
		check[x] = x;
		std::vector<int> &v1 = old_knn_vec[x];
		l1 = v1.size();
		for (i = 0; i < l1; ++i)
		{
			y = v1[i];
			check[y] = x;
			heap.push(std::make_pair(CalcDist(x, y), y));
			if (heap.size() == n_neighbors + 1) heap.pop();
		}
		for (i = 0; i < l1; ++i)
		{
			std::vector<int> &v2 = old_knn_vec[v1[i]];
			l2 = v2.size();
			for (j = 0; j < l2; ++j) if (check[y = v2[j]] != x)
			{
				check[y] = x;
				heap.push(std::make_pair(CalcDist(x, y), (int)y));
				if (heap.size() == n_neighbors + 1) heap.pop();
			}
		}
		while (!heap.empty())
		{
			knn_vec[x].push_back(heap.top().second);
			heap.pop();
		}
	}
	delete[] check;
}

void *LargeVis::propagation_thread_caller(void *arg)
{
	LargeVis *ptr = (LargeVis*)(((arg_struct*)arg)->ptr);
	ptr->propagation_thread(((arg_struct*)arg)->id);
	pthread_exit(NULL);
}

void LargeVis::run_propagation()
{
	for (int i = 0; i < n_propagations; ++i)
	{
		printf("LargeVis: Running Propagation %d/%lld%c\n", i + 1, n_propagations, 13);
		old_knn_vec = knn_vec;
		knn_vec = new std::vector<int>[n_vertices];
		pthread_t *pt = new pthread_t[n_threads];
		for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::propagation_thread_caller, new arg_struct(this, j));
		for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
		delete[] pt;
		delete[] old_knn_vec;
		old_knn_vec = NULL;
	}
	printf("LargeVis: Running Propagations Done\n");
}

void LargeVis::compute_similarity_thread(int id)
{
	long long lo = id * n_vertices / n_threads;
	long long hi = (id + 1) * n_vertices / n_threads;
	long long x, iter, p;
	real beta, lo_beta, hi_beta, sum_weight, H, tmp;
	for (x = lo; x < hi; ++x)
	{
	//	if(knn_not_changed[x]) continue;
	// FIXME: this can reduce time and error, but overall 
	// shape is bad and orphans make graph dirty
	// plus, time reduction is low so new method should be required

		beta = 1;
		lo_beta = hi_beta = -1;
		for (iter = 0; iter < 200; ++iter)
		{
			H = 0;
            sum_weight = FLT_MIN;
			for (p = head[x]; p >= 0; p = next[p])
			{
				sum_weight += tmp = exp(-beta * edge_weight[p]);
				H += beta * (edge_weight[p] * tmp);
			}
			H = (H / sum_weight) + log(sum_weight);
			if (fabs(H - log(perplexity)) < 1e-5) break;
			if (H > log(perplexity))
			{
				lo_beta = beta;
				if (hi_beta < 0) beta *= 2; else beta = (beta + hi_beta) / 2;
			}
			else
			{
				hi_beta = beta;
				if (lo_beta < 0) beta /= 2; else beta = (lo_beta + beta) / 2;
			}
            if(beta > FLT_MAX) beta = FLT_MAX;
        }
		for (p = head[x], sum_weight = FLT_MIN; p >= 0; p = next[p])
		{
			sum_weight += edge_weight[p] = exp(-beta * edge_weight[p]);

		}
		
		for (p = head[x]; p >= 0; p = next[p])
		{
			edge_weight[p] /= sum_weight;
		}

	}
}

void *LargeVis::compute_similarity_thread_caller(void *arg)
{
	LargeVis *ptr = (LargeVis*)(((arg_struct*)arg)->ptr);
	ptr->compute_similarity_thread(((arg_struct*)arg)->id);
	pthread_exit(NULL);
}

void LargeVis::search_reverse_thread(int id)
{
	long long lo = id * n_vertices / n_threads;
	long long hi = (id + 1) * n_vertices / n_threads;
	long long x, y, p, q;
	for (x = lo; x < hi; ++x)
	{
		for (p = head[x]; p >= 0; p = next[p])
		{
			y = edge_to[p];
			for (q = head[y]; q >= 0; q = next[q])
			{
				if (edge_to[q] == x) break;
			}
			reverse[p] = q;
		}
	}
}

void *LargeVis::search_reverse_thread_caller(void *arg)
{
	LargeVis *ptr = (LargeVis*)(((arg_struct*)arg)->ptr);
	ptr->search_reverse_thread(((arg_struct*)arg)->id);
	pthread_exit(NULL);
}

void LargeVis::compute_similarity()
{
    printf("LargeVis: Computing similarities ......\n");
	n_edge = 0;
	head = new long long[n_vertices];
	long long i, x, y, p, q;

	for (i = 0; i < n_vertices; ++i) head[i] = -1;
	for (x = 0; x < n_vertices; ++x)
	{
		for (i = 0; i < knn_vec[x].size(); ++i)
		{
			edge_from.push_back((int)x);
			edge_to.push_back((int)(y = knn_vec[x][i]));
			edge_weight.push_back(CalcDist(x, y));
			next.push_back(head[x]);
			reverse.push_back(-1);
			head[x] = n_edge++;
		}
	}
	
    //delete[] vec; vec = NULL;
    //delete[] knn_vec; knn_vec = NULL;
	pthread_t *pt = new pthread_t[n_threads];
	for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::compute_similarity_thread_caller, new arg_struct(this, j));
	for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
	delete[] pt;

	pt = new pthread_t[n_threads];
	for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::search_reverse_thread_caller, new arg_struct(this, j));
	for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
	delete[] pt;

	for (x = 0; x < n_vertices; ++x)
	{
		for (p = head[x]; p >= 0; p = next[p])
		{
			y = edge_to[p];
			q = reverse[p];
			if (q == -1)
			{
				edge_from.push_back((int)y);
				edge_to.push_back((int)x);
				edge_weight.push_back(0);
				next.push_back(head[y]);
				reverse.push_back(p);
				q = reverse[p] = head[y] = n_edge++;
			}
			if (x > y)
				edge_weight[p] = edge_weight[q] = (edge_weight[p] + edge_weight[q]) / 2;
			
		}
	}

	printf("LargeVis: Computing similarities Done.\n");
}

void LargeVis::test_accuracy()
{
	long long test_case = 100;
	std::priority_queue< pair<real, int> > *heap = new std::priority_queue< pair<real, int> >;
	long long hit_case = 0, i, j, x, y;
	for (i = 0; i < test_case; ++i)
	{
		x = floor(gsl_rng_uniform(gsl_r) * (n_vertices - 0.1));
		for (y = 0; y < n_vertices; ++y) if (x != y)
		{
			heap->push(std::make_pair(CalcDist(x, y), y));
			if (heap->size() == n_neighbors + 1) heap->pop();
		}
		while (!heap->empty())
		{
			y = heap->top().second;
			heap->pop();
			for (j = 0; j < knn_vec[x].size(); ++j) if (knn_vec[x][j] == y)
				++hit_case;
		}
	}
    delete heap;
	printf("LargeVis: Test knn accuracy : %.2f%%\n", hit_case * 100.0 / (test_case * n_neighbors));
}

void LargeVis::construt_knn()
{	
	sstart = clock();
	clock_gettime(CLOCK_MONOTONIC, &sstart_p);

	normalize();
	run_annoy();

	eend = clock();
	clock_gettime(CLOCK_MONOTONIC, &eend_p);

	real_time[0] = (double)(eend_p.tv_sec - sstart_p.tv_sec) + (double)(eend_p.tv_nsec - sstart_p.tv_nsec)/BILLION;
	ttt = eend - sstart;
	cpu_time[0] = (double)ttt / CLOCKS_PER_SEC;

	test_accuracy();

	if(knn_not_changed == NULL)
	{
		knn_not_changed = (bool *) malloc(n_vertices * sizeof(bool));
		for(int i = 0; i < n_vertices; ++i) knn_not_changed[i] = false;
	}

	sstart = clock();
	clock_gettime(CLOCK_MONOTONIC, &sstart_p);

	compute_similarity();
	
	eend = clock();
	clock_gettime(CLOCK_MONOTONIC, &eend_p);

	real_time[1] = (double)(eend_p.tv_sec - sstart_p.tv_sec) + (double)(eend_p.tv_nsec - sstart_p.tv_nsec)/BILLION;
	ttt = eend - sstart;
	cpu_time[1] = (double)ttt / CLOCKS_PER_SEC;

	printf("LargeVis: Normalize&RPTrees real time: %.2lf seconds!\n", real_time[0]);
	printf("LargeVis: Normalize&RPTrees clock time: %.2lf secs.\n", cpu_time[0]);
	printf("LargeVis: KNN to P real time: %.2lf seconds!\n", real_time[1]);
	printf("LargeVis: KNN to P clock time: %.2lf secs.\n", cpu_time[1]);
	printf("LargeVis: Construct_knn total real time: %.2lf seconds!\n", real_time[0] + real_time[1]);
	printf("LargeVis: Construct_knn total clock time: %.2lf secs.\n", cpu_time[0] + cpu_time[1]);
}

void LargeVis::get_result(unsigned long long** row_P, unsigned long long** col_P, double** val_P)
{

	long long size = edge_to.size();

	printf("LargeVis: knn edge size: %lld\n", size);

	*row_P = (unsigned long long*)malloc((n_vertices + 1) * sizeof(unsigned long long));
	*col_P = (unsigned long long*)malloc(size * sizeof(unsigned long long));
	*val_P = (double*)malloc(size * sizeof(double));

	printf("LargeVis: n_vertices: %lld, n_vertices*n_neighbors: %lld\n", n_vertices, n_vertices*n_neighbors);

	int need_log = 0;

	//FILE* fp_saved = fopen("saved.txt", "w+");
	//char temp_str[100] = "";

	long long x, y, p;
	long long i = 0;
	long long row_idx = 0;
	long long cnt = 0;
	(*row_P)[row_idx] = 0;
	for (x = 0; x < n_vertices; ++x)
	{
		
		for (p = head[x]; p >= 0; p = next[p])
		{
			
			(*col_P)[cnt] = edge_to[p];
			(*val_P)[cnt] = edge_weight[p];
			
			//printf("LargeVis: col_P[%lld]=%d\n", row_cnt-1, edge_to[p]);
			//printf("LargeVis: val_P[%lld]=%f\n", row_cnt-1, edge_weight[p]);
			// if (need_log){
			// 	sprintf(temp_str, "%lld %lld %f\n",(*row_P)[x], (*col_P)[cnt], (*val_P)[cnt]);
	  //       	fwrite(temp_str, strlen(temp_str), 1, fp_saved);
   //      	}

        	++cnt;
		}
		
		(*row_P)[x+1] = cnt;
	}

	//fclose(fp_saved);	

	printf("LargeVis: P Exported(get_result)\n");
}

void LargeVis::run(long long out_d, long long n_thre, long long n_samp, long long n_prop, real alph, long long n_tree, long long n_nega, long long n_neig, real gamm, real perp)
{
	gsl_rng_env_setup();
	gsl_T = gsl_rng_rand48;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);

	clean_model();
	if (!vec && !head)
	{
		printf("LargeVis: Missing training data!\n");
		return;
	}
	out_dim = out_d < 0 ? 2 : out_d;
	initial_alpha = alph < 0 ? 1.0 : alph;
	n_threads = n_thre < 0 ? 1 : n_thre;
	n_samples = n_samp;
	n_negatives = n_nega < 0 ? 5 : n_nega;
	n_neighbors = n_neig < 0 ? 150 : n_neig;
	n_trees = n_tree;
	n_propagations = n_prop < 0 ? 3 : n_prop;
	gamma = gamm < 0 ? 7.0 : gamm;
	perplexity = perp < 0 ? 50.0 : perp;

	if (n_samples < 0)
	{
		if (n_vertices < 10000)
			n_samples = 1000;
		else if (n_vertices < 1000000)
			n_samples = (n_vertices - 10000) * 9000 / (1000000 - 10000) + 1000;
		else n_samples = n_vertices / 100;
	}
	n_samples *= 1000000;
	if (n_trees < 0)
	{
		if (n_vertices < 100000)
			n_trees = 5;
		else if (n_vertices < 1000000)
			n_trees = 10;
		else if (n_vertices < 5000000)
			n_trees = 25;
		else n_trees = 50;
	}
	printf("LargeVis: Threads: %lld\n", n_threads);
	if (vec) { clean_graph(); construt_knn(); }
}

void LargeVis::run_propagation_once(int i, bool knn_validation)
{
	int cnt = 0;

	printf("LargeVis: Running propagation %d\n", i + 1);

	sstart = clock();
	clock_gettime(CLOCK_MONOTONIC, &sstart_p);
	
	old_knn_vec = knn_vec;
	knn_vec = new std::vector<int>[n_vertices];
	pthread_t *pt = new pthread_t[n_threads];
	for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::propagation_thread_caller, new arg_struct(this, j));
	for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
	delete[] pt;

	if(knn_validation)
	{
		for(int i = 0; i < n_vertices ; ++i)
		{
			if(old_knn_vec[i] == knn_vec[i])
			{
				knn_not_changed[i] = true;
				cnt++;
			}
			else
			{
				knn_not_changed[i] = false;
			}
		}
	}

	eend = clock();
	clock_gettime(CLOCK_MONOTONIC, &eend_p);

	real_time[i*2+2] = (double)(eend_p.tv_sec - sstart_p.tv_sec) + (double)(eend_p.tv_nsec - sstart_p.tv_nsec)/BILLION;
	ttt = eend - sstart;
	cpu_time[i*2+2] = (double)ttt / CLOCKS_PER_SEC;

	printf("LargeVis: Running propagation %d Done\n", i + 1);
	printf("LargeVis: KNN which will be skipped in propagation %d: %d\n", i+2, cnt);

	//TODO: Calculate and export compared data of old_knn_vec and knn_vec
	//to use it in sgd approximation

	delete[] old_knn_vec;
	old_knn_vec = NULL;
	test_accuracy();
	clean_graph();

	sstart = clock();
	clock_gettime(CLOCK_MONOTONIC, &sstart_p);

	compute_similarity();

	eend = clock();
	clock_gettime(CLOCK_MONOTONIC, &eend_p);

	real_time[i*2+3] = (double)(eend_p.tv_sec - sstart_p.tv_sec) + (double)(eend_p.tv_nsec - sstart_p.tv_nsec)/BILLION;
	ttt = eend - sstart;
	cpu_time[i*2+3] = (double)ttt / CLOCKS_PER_SEC;

	printf("LargeVis: Propagation %d: %.2lf real seconds!\n", i + 1, real_time[i*2+2]);
	printf("LargeVis: Propagation %d: %.2lf clock seconds!\n", i + 1, cpu_time[i*2+2]);
	printf("LargeVis: Calculate P for propagation %d: %.2lf real seconds!\n", i + 1, real_time[i*2+3]);
	printf("LargeVis: Calculate P for propagation %d: %.2lf clock seconds!\n", i + 1, cpu_time[i*2+3]);
	//FIXME: This should be ignored when pipelining is disabled.
	printf("LargeVis: Propagation %d total: %.2lf real seconds!\n", i + 1, real_time[i*2+2] + real_time[i*2+3]);
	printf("LargeVis: Propagation %d total: %.2lf clock seconds!\n", i + 1, cpu_time[i*2+2] + cpu_time[i*2+3]);

	//TODO: need to free memory of knn_not_changed, vec, knn_vec when last result is read;
}

double* LargeVis::get_real_time()
{
	return real_time;
}

double* LargeVis::get_clock_time()
{
	return cpu_time;
}

void LargeVis::setThreadsNum(long long i)
{
	n_threads = i;
}