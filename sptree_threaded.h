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


#ifndef SPTREE_H
#define SPTREE_H

using namespace std;


class Cell {

	unsigned long long dimension;
    double* corner;
    double* width;
    
    
public:
	Cell(unsigned long long inp_dimension);
	Cell(unsigned long long inp_dimension, double* inp_corner, double* inp_width);
    ~Cell();
    
	double getCorner(unsigned long long d);
	double getWidth(unsigned long long d);
	void setCorner(unsigned long long d, double val);
	void setWidth(unsigned long long d, double val);
    bool containsPoint(double point[]);
};


class SPTree
{
    
    // Fixed constants
	static const unsigned long long QT_NODE_CAPACITY = 1;

    // A buffer we use when doing force computations
    double* buff;
    
    // Properties of this node in the tree
    SPTree* parent;
	unsigned long long dimension;
    bool is_leaf;
	unsigned long long size;
	unsigned long long cum_size;
        
    // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
    Cell* boundary;
    
    // Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
    double* data;
    double* center_of_mass;
	unsigned long long index[QT_NODE_CAPACITY];
    
    // Children
    SPTree** children;
	unsigned long long no_children;
    
public:
	SPTree(unsigned long long D, double* inp_data, unsigned long long N);
	SPTree(unsigned long long D, double* inp_data, double* inp_corner, double* inp_width);
	SPTree(unsigned long long D, double* inp_data, unsigned long long N, double* inp_corner, double* inp_width);
	SPTree(SPTree* inp_parent, unsigned long long D, double* inp_data, unsigned long long N, double* inp_corner, double* inp_width);
	SPTree(SPTree* inp_parent, unsigned long long D, double* inp_data, double* inp_corner, double* inp_width);
    ~SPTree();
    void setData(double* inp_data);
    SPTree* getParent();
    void construct(Cell boundary);
	bool insert(unsigned long long new_index);
    void subdivide();
    bool isCorrect();
    void rebuildTree();
	void getAllIndices(unsigned long long* indices);
	unsigned long long getDepth();
    void computeNonEdgeForces(unsigned long long point_index, double theta, double neg_f[], double* sum_Q, double buff[]);
	void computeEdgeForces(unsigned long long* row_P, unsigned long long* col_P, double* val_P, long long N, double* pos_f, long long num_threads);
    void print();
    
private:
    void init(SPTree* inp_parent, unsigned long long D, double* inp_data, double* inp_corner, double* inp_width);
    void fill(unsigned long long N);
    unsigned long long getAllIndices(unsigned long long* indices, unsigned long long loc);
    bool isChild(unsigned long long test_index, unsigned long long start, unsigned long long end);
};

#endif
