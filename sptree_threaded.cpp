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
#include <cmath>
#include "sptree.h"

#include <boost/thread.hpp>
#include <boost/random.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

long long global_N, global_num_threads, global_num_vertices, global_dimension;
unsigned long long *global_row_P, *global_col_P;
double *global_val_P, *global_data, *global_pos_f;

// Constructs cell
Cell::Cell(unsigned long long inp_dimension) {
    dimension = inp_dimension;
    corner = (double*) malloc(dimension * sizeof(double));
    width  = (double*) malloc(dimension * sizeof(double));
}

Cell::Cell(unsigned long long inp_dimension, double* inp_corner, double* inp_width) {
    dimension = inp_dimension;
    corner = (double*) malloc(dimension * sizeof(double));
    width  = (double*) malloc(dimension * sizeof(double));
    for(long long d = 0; d < dimension; d++) setCorner(d, inp_corner[d]);
    for(long long d = 0; d < dimension; d++) setWidth( d,  inp_width[d]);
}

// Destructs cell
Cell::~Cell() {
    free(corner);
    free(width);
}

double Cell::getCorner(unsigned long long d) {
    return corner[d];
}

double Cell::getWidth(unsigned long long d) {
    return width[d];
}

void Cell::setCorner(unsigned long long d, double val) {
    corner[d] = val;
}

void Cell::setWidth(unsigned long long d, double val) {
    width[d] = val;
}

// Checks whether a point lies in a cell
bool Cell::containsPoint(double point[])
{
    for(long long d = 0; d < dimension; d++) {
        if(corner[d] - width[d] > point[d]) return false;
        if(corner[d] + width[d] < point[d]) return false;
    }
    return true;
}


// Default constructor for SPTree -- build tree, too!
SPTree::SPTree(unsigned long long D, double* inp_data, unsigned long long N)
{
    
    // Compute mean, width, and height of current map (boundaries of SPTree)
    long long nD = 0;
    double* mean_Y = (double*) calloc(D,  sizeof(double));
    double*  min_Y = (double*) malloc(D * sizeof(double)); for(unsigned long long d = 0; d < D; d++)  min_Y[d] =  DBL_MAX;
    double*  max_Y = (double*) malloc(D * sizeof(double)); for(unsigned long long d = 0; d < D; d++)  max_Y[d] = -DBL_MAX;
    for(unsigned long long n = 0; n < N; n++) {
        for(unsigned long long d = 0; d < D; d++) {
            mean_Y[d] += inp_data[n * D + d];
            if(inp_data[nD + d] < min_Y[d]) min_Y[d] = inp_data[nD + d];
            if(inp_data[nD + d] > max_Y[d]) max_Y[d] = inp_data[nD + d];
        }
        nD += D;
    }
    for(long long d = 0; d < D; d++) mean_Y[d] /= (double) N;
    
    // Construct SPTree
    double* width = (double*) malloc(D * sizeof(double));
    for(long long d = 0; d < D; d++) width[d] = fmax(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
    init(NULL, D, inp_data, mean_Y, width);
    fill(N);
    
    // Clean up memory
    free(mean_Y);
    free(max_Y);
    free(min_Y);
    free(width);
}


// Constructor for SPTree with particular size and parent -- build the tree, too!
SPTree::SPTree(unsigned long long D, double* inp_data, unsigned long long N, double* inp_corner, double* inp_width)
{
    init(NULL, D, inp_data, inp_corner, inp_width);
    fill(N);
}


// Constructor for SPTree with particular size (do not fill the tree)
SPTree::SPTree(unsigned long long D, double* inp_data, double* inp_corner, double* inp_width)
{
    init(NULL, D, inp_data, inp_corner, inp_width);
}


// Constructor for SPTree with particular size and parent (do not fill tree)
SPTree::SPTree(SPTree* inp_parent, unsigned long long D, double* inp_data, double* inp_corner, double* inp_width) {
    init(inp_parent, D, inp_data, inp_corner, inp_width);
}


// Constructor for SPTree with particular size and parent -- build the tree, too!
SPTree::SPTree(SPTree* inp_parent, unsigned long long D, double* inp_data, unsigned long long N, double* inp_corner, double* inp_width)
{
    init(inp_parent, D, inp_data, inp_corner, inp_width);
    fill(N);
}


// Main initialization function
void SPTree::init(SPTree* inp_parent, unsigned long long D, double* inp_data, double* inp_corner, double* inp_width)
{
    parent = inp_parent;
    dimension = D;
    no_children = 2;
    for(unsigned long long d = 1; d < D; d++) no_children *= 2;
    data = inp_data;
    is_leaf = true;
    size = 0;
    cum_size = 0;
    
    boundary = new Cell(dimension);
    for(unsigned long long d = 0; d < D; d++) boundary->setCorner(d, inp_corner[d]);
    for(unsigned long long d = 0; d < D; d++) boundary->setWidth( d, inp_width[d]);
    
    children = (SPTree**) malloc(no_children * sizeof(SPTree*));
    for(unsigned long long i = 0; i < no_children; i++) children[i] = NULL;

    center_of_mass = (double*) malloc(D * sizeof(double));
    for(unsigned long long d = 0; d < D; d++) center_of_mass[d] = .0;
    
    buff = (double*) malloc(D * sizeof(double));
}


// Destructor for SPTree
SPTree::~SPTree()
{
    for(unsigned long long i = 0; i < no_children; i++) {
        if(children[i] != NULL) delete children[i];
    }
    free(children);
    free(center_of_mass);
    free(buff);
    delete boundary;
}


// Update the data underlying this tree
void SPTree::setData(double* inp_data)
{
    data = inp_data;
}


// Get the parent of the current tree
SPTree* SPTree::getParent()
{
    return parent;
}


// Insert a point into the SPTree
bool SPTree::insert(unsigned long long new_index)
{
    // Ignore objects which do not belong in this quad tree
    double* point = data + new_index * dimension;
    if(!boundary->containsPoint(point))
        return false;
    
    // Online update of cumulative size and center-of-mass
    cum_size++;
    double mult1 = (double) (cum_size - 1) / (double) cum_size;
    double mult2 = 1.0 / (double) cum_size;
    for(unsigned long long d = 0; d < dimension; d++) center_of_mass[d] *= mult1;
    for(unsigned long long d = 0; d < dimension; d++) center_of_mass[d] += mult2 * point[d];
    
    // If there is space in this quad tree and it is a leaf, add the object here
    if(is_leaf && size < QT_NODE_CAPACITY) {
        index[size] = new_index;
        size++;
        return true;
    }
    
    // Don't add duplicates for now (this is not very nice)
    bool any_duplicate = false;
    for(unsigned long long n = 0; n < size; n++) {
        bool duplicate = true;
        for(unsigned long long d = 0; d < dimension; d++) {
            if(point[d] != data[index[n] * dimension + d]) { duplicate = false; break; }
        }
        any_duplicate = any_duplicate | duplicate;
    }
    if(any_duplicate) return true;
    
    // Otherwise, we need to subdivide the current cell
    if(is_leaf) subdivide();
    
    // Find out where the point can be inserted
    for(unsigned long long i = 0; i < no_children; i++) {
        if(children[i]->insert(new_index)) return true;
    }
    
    // Otherwise, the point cannot be inserted (this should never happen)
    return false;
}

    
// Create four children which fully divide this cell into four quads of equal area
void SPTree::subdivide() {
    
    // Create new children
    double* new_corner = (double*) malloc(dimension * sizeof(double));
    double* new_width  = (double*) malloc(dimension * sizeof(double));
    for(unsigned long long i = 0; i < no_children; i++) {
        unsigned long long div = 1;
        for(unsigned long long d = 0; d < dimension; d++) {
            new_width[d] = .5 * boundary->getWidth(d);
            if((i / div) % 2 == 1) new_corner[d] = boundary->getCorner(d) - .5 * boundary->getWidth(d);
            else                   new_corner[d] = boundary->getCorner(d) + .5 * boundary->getWidth(d);
            div *= 2;
        }
        children[i] = new SPTree(this, dimension, data, new_corner, new_width);
    }
    free(new_corner);
    free(new_width);
    
    // Move existing points to correct children
    for(unsigned long long i = 0; i < size; i++) {
        bool success = false;
        for(unsigned long long j = 0; j < no_children; j++) {
            if(!success) success = children[j]->insert(index[i]);
        }
        index[i] = -1;
    }
    
    // Empty parent node
    size = 0;
    is_leaf = false;
}


// Build SPTree on dataset
void SPTree::fill(unsigned long long N)
{
    for(unsigned long long i = 0; i < N; i++) insert(i);
}


// Checks whether the specified tree is correct
bool SPTree::isCorrect()
{
    for(unsigned long long n = 0; n < size; n++) {
        double* point = data + index[n] * dimension;
        if(!boundary->containsPoint(point)) return false;
    }
    if(!is_leaf) {
        bool correct = true;
        for(long long i = 0; i < no_children; i++) correct = correct && children[i]->isCorrect();
        return correct;
    }
    else return true;
}



// Build a list of all indices in SPTree
void SPTree::getAllIndices(unsigned long long* indices)
{
    getAllIndices(indices, 0);
}


// Build a list of all indices in SPTree
unsigned long long SPTree::getAllIndices(unsigned long long* indices, unsigned long long loc)
{
    
    // Gather indices in current quadrant
    for(unsigned long long i = 0; i < size; i++) indices[loc + i] = index[i];
    loc += size;
    
    // Gather indices in children
    if(!is_leaf) {
        for(long long i = 0; i < no_children; i++) loc = children[i]->getAllIndices(indices, loc);
    }
    return loc;
}


unsigned long long SPTree::getDepth() {
    if(is_leaf) return 1;
    long long depth = 0;
    for(unsigned long long i = 0; i < no_children; i++) depth = fmax(depth, children[i]->getDepth());
    return 1 + depth;
}


// Compute non-edge forces using Barnes-Hut algorithm
void SPTree::computeNonEdgeForces(unsigned long long point_index, double theta, double neg_f[], double* sum_Q, double buff[])
{
    // Make sure that we spend no time on empty nodes or self-interactions
    if(cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return;
    
    // Compute distance between point and center-of-mass
    double D = .0;
    unsigned long long ind = point_index * dimension;
    for(unsigned long long d = 0; d < dimension; d++) buff[d] = data[ind + d] - center_of_mass[d];
    for(unsigned long long d = 0; d < dimension; d++) D += buff[d] * buff[d];
    
    // Check whether we can use this node as a "summary"
    double max_width = 0.0;
    double cur_width;
    for(unsigned long long d = 0; d < dimension; d++) {
        cur_width = boundary->getWidth(d);
        max_width = (max_width > cur_width) ? max_width : cur_width;
    }
    if(is_leaf || max_width / sqrt(D) < theta) {
    
        // Compute and add t-SNE force between point and current node
        D = 1.0 / (1.0 + D);
        double mult = cum_size * D;
        *sum_Q += mult;
        mult *= D;
        for(unsigned long long d = 0; d < dimension; d++) neg_f[d] += mult * buff[d];
    }
    else {

        // Recursively apply Barnes-Hut to children
        for(unsigned long long i = 0; i < no_children; i++) children[i]->computeNonEdgeForces(point_index, theta, neg_f, sum_Q, buff);
    }
}

void *computeEdgeForcesThread(void *_id)
{
	long long id = (long long)_id;
	long long lo = id * global_num_vertices / global_num_threads;
	long long hi = (id + 1) * global_num_vertices / global_num_threads;

	// Loop over all edges in the graph
	unsigned long long ind1 = lo * global_dimension;
	unsigned long long ind2 = 0;
	double D;
	for (unsigned long long n = lo; n < hi; n++) {
		for (unsigned long long i = global_row_P[n]; i < global_row_P[n + 1]; i++) {

			// Compute pairwise distance and Q-value
			D = 1.0;
			ind2 = global_col_P[i] * global_dimension;
			for (unsigned long long d = 0; d < global_dimension; d++) D += (global_data[ind1 + d] - global_data[ind2 + d]) * (global_data[ind1 + d] - global_data[ind2 + d]);
			D = global_val_P[i] / D;

			// Sum positive force
			for (unsigned long long d = 0; d < global_dimension; d++) global_pos_f[ind1 + d] += D * (global_data[ind1 + d] - global_data[ind2 + d]);
		}
		ind1 += global_dimension;
	}

	return NULL;
}

// Computes edge forces
void SPTree::computeEdgeForces(unsigned long long* row_P, unsigned long long* col_P, double* val_P, long long N, double* pos_f, long long num_threads)
{
	global_N = N;
	global_num_threads = num_threads;
	global_num_vertices = N;
	global_dimension = dimension;
	global_row_P = row_P;
	global_col_P = col_P;
	global_val_P = val_P;
	global_data = data;
	global_pos_f = pos_f;

	boost::thread *pt = new boost::thread[num_threads];
	for (long long i = 0; i < num_threads; ++i) pt[i] = boost::thread(computeEdgeForcesThread, (void*)i);
	for (long long i = 0; i < num_threads; ++i) pt[i].join();
	delete[] pt;
}


// Print out tree
void SPTree::print() 
{
    if(cum_size == 0) {
        printf("Empty node\n");
        return;
    }

    if(is_leaf) {
        printf("Leaf node; data = [");
        for(long long i = 0; i < size; i++) {
            double* point = data + index[i] * dimension;
            for(long long d = 0; d < dimension; d++) printf("%f, ", point[d]);
            printf(" (index = %lld)", index[i]);
            if(i < size - 1) printf("\n");
            else printf("]\n");
        }        
    }
    else {
        printf("Intersection node with center-of-mass = [");
        for(long long d = 0; d < dimension; d++) printf("%f, ", center_of_mass[d]);
        printf("]; children are:\n");
        for(long long i = 0; i < no_children; i++) children[i]->print();
    }
}

