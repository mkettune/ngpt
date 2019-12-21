/*
 *  Copyright (c) 2016, Marco Manzi and Markus Kettunen. All rights reserved.
 *  Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     *  Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *     *  Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *     *  Neither the name of the NVIDIA CORPORATION nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Backend.hpp"
#include <malloc.h>
#include <stdio.h>

#include <windows.h>
#undef min
#undef max

using namespace poisson;

//------------------------------------------------------------------------

Backend::Backend(void)
{
    LARGE_INTEGER freq;
    if (!QueryPerformanceFrequency(&freq))
    {
        printf("QueryPerformanceFrequency() failed!\n");
        exit(0);
    }
    m_timerTicksToSecs = max(1.0 / (double)freq.QuadPart, 0.0);
}

//------------------------------------------------------------------------

Backend::~Backend(void)
{
}

//------------------------------------------------------------------------

Backend::Vector* Backend::allocVector(int numElems, size_t bytesPerElem)
{
    assert(numElems >= 0);
    assert(bytesPerElem > 0);

    Vector* x       = new Vector;
    x->numElems     = numElems;
    x->bytesPerElem = bytesPerElem;
    x->bytesTotal   = numElems * bytesPerElem;
    x->ptr          = malloc(x->bytesTotal);

    if (!x->ptr)
        fail("Out of memory!");
    return x;
}

//------------------------------------------------------------------------

void Backend::freeVector(Vector* x)
{
    if (x && x->ptr)
        free(x->ptr);
    delete x;
}

//------------------------------------------------------------------------

void* Backend::map(Vector* x)
{
    assert(x);
    return x->ptr;
}

//------------------------------------------------------------------------

void Backend::unmap(Vector* x, void* ptr, bool modified)
{
    (void)x;
    (void)ptr;
    (void)modified;
}

//------------------------------------------------------------------------

void Backend::set(Vector* x, float y)
{
    assert(x && x->bytesPerElem % sizeof(float) == 0);

    float*  p_x = (float*)map(x);
    int     n   = (int)(x->bytesTotal / sizeof(float));

    for (int i = 0; i < n; i++)
        p_x[i] = y;

    unmap(x, (void*)p_x, true);
}

//------------------------------------------------------------------------

void Backend::copy(Vector* x, Vector* y)
{
    assert(x && y && x->bytesTotal == y->bytesTotal);

    float*          p_x = (float*)map(x);
    const float*    p_y = (const float*)map(y);
    int             n   = (int)(x->bytesTotal / sizeof(float));

    for (int i = 0; i < n; i++)
        p_x[i] = p_y[i];

    unmap(x, (void*)p_x, true);
    unmap(y, (void*)p_y, false);
}

//------------------------------------------------------------------------

void Backend::read(void* ptr, Vector* x)
{
    assert(ptr && x);
    void* p_x = map(x);
    memcpy(ptr, p_x, x->bytesTotal);
    unmap(x, p_x, false);
}

//------------------------------------------------------------------------

void Backend::write(Vector* x, const void* ptr)
{
    assert(x && ptr);
    void* p_x = map(x);
    memcpy(p_x, ptr, x->bytesTotal);
    unmap(x, p_x, true);
}

//=================== FOR COMMENTS SEE CUDA CODE =========================
// Compute P*x 
//
void Backend::calc_Px(Vector* Px, PoissonMatrix P, Vector* x, int nframes, Vector* mv)
{
	int nimg = P.n_constr;
	
	assert(Px && Px->numElems == P.size.x * P.size.y * nimg * nframes && Px->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
    assert(x && x->numElems == P.size.x * P.size.y * nframes && x->bytesPerElem == sizeof(Vec3f));
	

	Vec3f* p_mv;
	if (P.useMotion)
		p_mv     = (Vec3f*)map(mv);

    Vec3f*          p_Px    = (Vec3f*)map(Px);
    const Vec3f*    p_x     = (const Vec3f*)map(x);
    int             n       = P.size.x * P.size.y;

	for (int yy = 0, i = 0; yy < P.size.y; yy++)
	for (int xx = 0; xx < P.size.x; xx++, i++)
	{
		for (int f=0; f < nframes; f++){
			Vec3f xi = p_x[f*n + i];

			int fo = nimg*f*n;

			p_Px[fo + n * 0 + i] = xi * P.alpha;
			p_Px[fo + n * 1 + i] = (xx != P.size.x - 1) ? p_x[f*n + i + 1] - xi : 0.0f;
			p_Px[fo + n * 2 + i] = (yy != P.size.y - 1) ? p_x[f*n + i + P.size.x] - xi : 0.0f;

			if (P.useTime){
				if (P.useMotion){
					Vec3f to = p_mv[f*n + i]; //correct
					int offset = to.y*P.size.x + to.x;
					bool valid = (f != nframes - 1) && (xx < P.size.x - to.x) && (xx >= to.x) && (yy < P.size.y - to.y) && (yy >= to.y);

					p_Px[fo + n * P.dt_idx + i] = valid ? P.w_dt * (p_x[f*n + i + n + offset] - xi) : 0.0f;

					if (P.useDDxt){
						bool valid_x = (f != nframes - 1) && (xx < P.size.x - to.x - 1) && (xx >= to.x) && (yy < P.size.y - to.y)	 && (yy >= to.y);			
						bool valid_y = (f != nframes - 1) && (xx < P.size.x - to.x)		&& (xx >= to.x) && (yy < P.size.y - to.y -1) && (yy >= to.y);

						p_Px[fo + n * P.dxdt_idx + i]	= valid_x ? P.w_dsdt*(p_x[f*n + i + n + 1 + offset]			- p_x[f*n + i + n + offset] - p_x[f*n + i + 1] + xi) : 0.0f;
						p_Px[fo + n * P.dydt_idx + i]	= valid_y ? P.w_dsdt*(p_x[f*n + i + n + P.size.x + offset]	- p_x[f*n + i + n + offset] - p_x[f*n + i + P.size.x] + xi) : 0.0f;
					}
				}
				else{
					p_Px[fo + n * P.dt_idx + i] = (f != nframes - 1) ? P.w_dt*(p_x[f*n + i + n] - xi) : 0.0f;

					if (P.useDDxt){
						p_Px[fo + n * P.dxdt_idx + i]	= (f != nframes - 1 && xx != P.size.x - 1) ? (p_x[f*n + i + n + 1] - p_x[f*n + i + n] - p_x[f*n + i + 1] + xi)*P.w_dsdt : 0.0f;
						p_Px[fo + n * P.dydt_idx + i]	= (f != nframes - 1 && yy != P.size.y - 1) ? (p_x[f*n + i + n + P.size.x] - p_x[f*n + i + n] - p_x[f*n + i + P.size.x] + xi)*P.w_dsdt : 0.0f;

					}
				}
			}
		}

	}

    unmap(Px,   (void*)p_Px, true);
    unmap(x,    (void*)p_x,  false);
}

//=================== FOR COMMENTS SEE CUDA CODE =========================
// Compute (P'*W^2)*x. Note that P is transposed. 
//
void Backend::calc_PTW2x(Vector* PTW2x, PoissonMatrix P, Vector* w2, Vector* x, int nframes, Vector* mv, Vector* mv_bw)
{
	int nimg = P.n_constr;

	assert(PTW2x && PTW2x->numElems == P.size.x * P.size.y * nframes && PTW2x->bytesPerElem == sizeof(Vec3f));
	assert(P.size.x >= 0 && P.size.y >= 0);
	assert(w2 && w2->numElems == P.size.x * P.size.y * nimg * nframes && w2->bytesPerElem == sizeof(float));
	assert(x && x->numElems == P.size.x * P.size.y * nimg * nframes && x->bytesPerElem == sizeof(Vec3f));


	Vec3f* p_mv;
	Vec3f* p_mv_bw;
	if (P.useMotion){
		p_mv     = (Vec3f*)map(mv);
		p_mv_bw  = (Vec3f*)map(mv_bw);
	}
    Vec3f*          p_PTW2x = (Vec3f*)map(PTW2x);
    const float*    p_w2    = (const float*)map(w2);
    const Vec3f*    p_x     = (const Vec3f*)map(x);
    int             n       = P.size.x * P.size.y;

	for (int yy = 0, i = 0; yy < P.size.y; yy++)
	for (int xx = 0; xx < P.size.x; xx++, i++)
	{
		for (int f=0; f < nframes; f++){
			int fo = nimg* n*f;
			Vec3f PTW2xi					= p_w2[fo + n * 0 + i]				* p_x[fo + n * 0 + i] * P.alpha;
			if (xx != 0)            PTW2xi += p_w2[fo + n * 1 + i - 1]			* p_x[fo + n * 1 + i - 1];
			if (xx != P.size.x - 1) PTW2xi -= p_w2[fo + n * 1 + i]				* p_x[fo + n * 1 + i];
			if (yy != 0)            PTW2xi += p_w2[fo + n * 2 + i - P.size.x]	* p_x[fo + n * 2 + i - P.size.x];
			if (yy != P.size.y - 1) PTW2xi -= p_w2[fo + n * 2 + i]				* p_x[fo + n * 2 + i];

			if (P.useTime){
				if (P.useMotion){
					Vec3f to	= p_mv[f*n + i];
					Vec3f to_bw = (f != 0) ? p_mv_bw[f*n + i - n] : Vec3f(0);

					int offset = to.y*P.size.x + to.x;
					int offset_bw = to_bw.y*P.size.x + to_bw.x;

					bool valid		= (f != nframes -1) && (xx < P.size.x - to.x)	 && (xx >= to.x)	&& (yy < P.size.y - to.y)	 && (yy >= to.y);
					bool valid_bw	= (f != 0)			&& (xx < P.size.x - to_bw.x) && (xx >= to_bw.x) && (yy < P.size.y - to_bw.y) && (yy >= to_bw.y);

					if (valid)		PTW2xi -=  p_w2[fo + P.dt_idx * n + i]								* p_x[fo + P.dt_idx * n + i]								* P.w_dt;
					if (valid_bw)	PTW2xi +=  p_w2[fo + P.dt_idx * n + i + offset_bw - P.n_constr * n] * p_x[fo + P.dt_idx * n + i + offset_bw - P.n_constr * n]	* P.w_dt;

					if (P.useDDxt){
						Vec3f toX	 = (xx != 0) ? p_mv[f*n + i - 1] : Vec3f(0);
						Vec3f to_bwX = (f != 0 && xx != 0) ? p_mv_bw[f*n + i - 1 - n] : Vec3f(0);
						Vec3f toY	 = (yy != 0) ? p_mv[f*n + i - P.size.x] : Vec3f(0);
						Vec3f to_bwY = (f != 0 && yy != 0) ? p_mv_bw[f*n + i - P.size.x - n] : Vec3f(0);


						int offset_bwX = to_bwX.y*P.size.x + to_bwX.x;	//backwards offset is already negated!
						int offset_bwY = to_bwY.y*P.size.x + to_bwY.x;	//backwards offset is already negated!

						// check if offset leads to valid pixel
						bool valid1 = (f != nframes - 1) && (xx < P.size.x - to.x - 1)	  && (xx >= to.x)		  && (yy < P.size.y - to.y)     && (yy >= to.y);
						bool valid2 = (f != nframes - 1) && (xx < P.size.x - toX.x)		  && (xx >= toX.x + 1)	  && (yy < P.size.y - toX.y)    && (yy >= toX.y);
						bool valid3 = (f != 0)		     && (xx < P.size.x - to_bw.x - 1) && (xx >= to_bw.x)	  && (yy < P.size.y - to_bw.y)  && (yy >= to_bw.y);
						bool valid4 = (f != 0)			 && (xx < P.size.x - to_bwX.x)	  && (xx >= to_bwX.x + 1) && (yy < P.size.y - to_bwX.y) && (yy >= to_bwX.y);

						if (valid1)	PTW2xi +=  p_w2[fo + P.dxdt_idx*n + i]									 * p_x[fo + P.dxdt_idx*n + i]									* P.w_dsdt;
						////if (valid2)	PTW2xi -=  p_w2[fo + P.dxdt_idx*n + i - 1]								 * p_x[fo + P.dxdt_idx*n + i - 1]								* P.w_dsdt;
						if (valid3)	PTW2xi -=  p_w2[fo + P.dxdt_idx*n + i - P.n_constr * n + offset_bw]		 * p_x[fo + P.dxdt_idx*n + i - P.n_constr*n + offset_bw]		* P.w_dsdt;
						if (valid4)	PTW2xi +=  p_w2[fo + P.dxdt_idx*n + i - 1 - P.n_constr * n + offset_bwX] * p_x[fo + P.dxdt_idx*n + i - 1 - P.n_constr*n + offset_bwX]	* P.w_dsdt;

						valid1 = (f != nframes - 1) && (xx < P.size.x - to.x)	  && (xx >= to.x)		&& (yy < P.size.y - to.y - 1)	 && (yy >= to.y);
						valid2 = (f != nframes - 1) && (xx < P.size.x - toY.x)	  && (xx >= toY.x)		&& (yy < P.size.y - toY.y)		 && (yy >= toY.y + 1);
						valid3 = (f != 0)			&& (xx < P.size.x - to_bw.x)  && (xx >= to_bw.x)	&& (yy < P.size.y - to_bw.y - 1) && (yy >= to_bw.y);
						valid4 = (f != 0)			&& (xx < P.size.x - to_bwY.x) && (xx >= to_bwY.x)	&& (yy < P.size.y - to_bwY.y)	 && (yy >= to_bwY.y + 1);

						if (valid1)	PTW2xi +=  p_w2[fo + P.dydt_idx*n + i]											* p_x[fo + P.dydt_idx*n + i]										* P.w_dsdt;
						if (valid2)	PTW2xi -=  p_w2[fo + P.dydt_idx*n + i - P.size.x]								* p_x[fo + P.dydt_idx*n + i - P.size.x]								* P.w_dsdt;
						if (valid3)	PTW2xi -=  p_w2[fo + P.dydt_idx*n + i - P.n_constr * n + offset_bw]				* p_x[fo + P.dydt_idx*n + i - P.n_constr*n + offset_bw]				* P.w_dsdt;
						if (valid4)	PTW2xi +=  p_w2[fo + P.dydt_idx*n + i - P.size.x - P.n_constr * n + offset_bwY] * p_x[fo + P.dydt_idx*n + i - P.size.x - P.n_constr*n + offset_bwY] * P.w_dsdt;
					}
				}
				else{
					if (f != nframes - 1)	PTW2xi  -= p_w2[fo + P.dt_idx * n + i] * p_x[fo + P.dt_idx * n + i] * P.w_dt;
					if (f != 0)				PTW2xi  += p_w2[fo + P.dt_idx * n + i - P.n_constr * n] * p_x[fo + P.dt_idx * n + i - P.n_constr * n] * P.w_dt;

					if (P.useDDxt){
						//A^t W^2 x (x,y,f) = all stuff from above PLUS
						if (f != nframes - 1 && xx != P.size.x - 1) PTW2xi +=  p_w2[fo + P.dxdt_idx * n + i] * p_x[fo + P.dxdt_idx * n + i] * P.w_dsdt;
						if (f != nframes - 1 && xx != 0)			PTW2xi -=  p_w2[fo + P.dxdt_idx * n + i - 1] * p_x[fo + P.dxdt_idx * n + i - 1] * P.w_dsdt;
						if (f != 0 && xx != P.size.x - 1)			PTW2xi -=  p_w2[fo + P.dxdt_idx * n + i - P.n_constr * n] * p_x[fo + P.dxdt_idx * n + i - P.n_constr * n] * P.w_dsdt;
						if (f != 0 && xx != 0)						PTW2xi +=  p_w2[fo + P.dxdt_idx * n + i - 1 - P.n_constr * n] * p_x[fo + P.dxdt_idx * n + i - 1 - P.n_constr * n] * P.w_dsdt;

						if (f != nframes - 1 && yy != P.size.y - 1) PTW2xi +=  p_w2[fo + P.dydt_idx * n + i] * p_x[fo + P.dydt_idx * n + i] * P.w_dsdt;
						if (f != nframes - 1 && yy != 0)			PTW2xi -=  p_w2[fo + P.dydt_idx * n + i - P.size.x] * p_x[fo + P.dydt_idx * n + i - P.size.x] * P.w_dsdt;
						if (f != 0 && yy != P.size.y - 1)			PTW2xi -=  p_w2[fo + P.dydt_idx * n + i - P.n_constr * n] * p_x[fo + P.dydt_idx * n + i - P.n_constr * n] * P.w_dsdt;
						if (f != 0 && yy != 0)						PTW2xi +=  p_w2[fo + P.dydt_idx * n + i - P.size.x - P.n_constr * n] * p_x[fo + P.dydt_idx * n + i - P.size.x - P.n_constr * n] * P.w_dsdt;
					}
				}
			}
			p_PTW2x[n*f + i] = PTW2xi;
		}
	}
    unmap(PTW2x,    (void*)p_PTW2x, true);
    unmap(w2,       (void*)p_w2,    false);
    unmap(x,        (void*)p_x,     false);
}

//=================== FOR COMMENTS SEE CUDA CODE =========================
// Compute Ax = (P'*W^2*P)*x and xAx = x'*(P'*W^2*P)*x
// 
void Backend::calc_Ax_xAx(Vector* Ax, Vector* xAx, PoissonMatrix P, Vector* w2, Vector* x, int nframes, Vector* mv, Vector* mv_bw)
{
	int nimg = P.n_constr;

	assert(Ax && Ax->numElems == P.size.x * P.size.y * nframes && Ax->bytesPerElem == sizeof(Vec3f));
	assert(xAx && xAx->numElems == 1 && xAx->bytesPerElem == sizeof(Vec3f));
	assert(P.size.x >= 0 && P.size.y >= 0);
	assert(w2 && w2->numElems == P.size.x * P.size.y * nimg * nframes && w2->bytesPerElem == sizeof(float));
	assert(x && x->numElems == P.size.x * P.size.y * nframes && x->bytesPerElem == sizeof(Vec3f));

	Vec3f* p_mv;
	Vec3f* p_mv_bw;
	if (P.useMotion){
		p_mv     = (Vec3f*)map(mv);
		p_mv_bw  = (Vec3f*)map(mv_bw);
	}
	Vec3f*          p_Ax        = (Vec3f*)map(Ax); 
	Vec3f*          p_xAx       = (Vec3f*)map(xAx); 
	const float*    p_w2        = (const float*)map(w2);
	const Vec3f*    p_x         = (const Vec3f*)map(x); 
    int             n           = P.size.x * P.size.y;
    float           alphaSqr    = P.alpha * P.alpha;

    p_xAx[0] = 0.0f;
    for (int yy = 0, i = 0; yy < P.size.y; yy++)
	for (int xx = 0; xx < P.size.x; xx++, i++)
	{
		for (int f=0; f < nframes; f++){
			Vec3f xi = p_x[f*n + i];

			int fo = nimg*f*n;
			Vec3f Axi					 = p_w2[fo + n * 0 + i]				* xi * alphaSqr;
			if (xx != 0)            Axi += p_w2[fo + n * 1 + i - 1]			* (xi - p_x[f*n + i - 1]);
			if (xx != P.size.x - 1) Axi += p_w2[fo + n * 1 + i]				* (xi - p_x[f*n + i + 1]);
			if (yy != 0)            Axi += p_w2[fo + n * 2 + i - P.size.x]	* (xi - p_x[f*n + i - P.size.x]);
			if (yy != P.size.y - 1) Axi += p_w2[fo + n * 2 + i]				* (xi - p_x[f*n + i + P.size.x]);

			if (P.useTime){
				float w_dt2 = P.w_dt*P.w_dt;
				if (P.useMotion){
					//get forward and backward motion vector
					Vec3f to = p_mv[f*n + i];
					Vec3f to_bw = (f != 0) ? p_mv_bw[f*n + i - n] : Vec3f(0);
					//Vec3f to_bw = p_mv_bw[f*n + i];

					// compute offsets
					int offset = to.y*P.size.x + to.x;
					int offset_bw = to_bw.y*P.size.x + to_bw.x;		//backwards offset is already negated!

					// check if offset leads to valid path
					bool valid	  = (f != nframes - 1)	&& (xx < P.size.x - to.x)	 && (xx >= to.x)	&& (yy < P.size.y - to.y)	 && (yy >= to.y);
					bool valid_bw = (f != 0)			&& (xx < P.size.x - to_bw.x) && (xx >= to_bw.x) && (yy < P.size.y - to_bw.y) && (yy >= to_bw.y);

					if (valid)		Axi +=  p_w2[fo + P.dt_idx * n + i]								 * (xi - p_x[f*n + i + n + offset])	   * w_dt2;
					if (valid_bw)	Axi +=  p_w2[fo + P.dt_idx * n + i + offset_bw - P.n_constr * n] * (xi - p_x[f*n + i - n + offset_bw]) * w_dt2;

					if (P.useDDxt){
						float w_dsdt2 = P.w_dsdt*P.w_dsdt;


						Vec3f toX	 = (xx != 0) ? p_mv[f*n + i - 1] : Vec3f(0);
						Vec3f to_bwX = (f != 0 && xx != 0) ? p_mv_bw[f*n + i - 1 - n] : Vec3f(0);
						Vec3f toY	 = (yy != 0) ? p_mv[f*n + i - P.size.x] : Vec3f(0);
						Vec3f to_bwY = (f != 0 && yy != 0) ? p_mv_bw[f*n + i - P.size.x - n] : Vec3f(0);

						// compute offset
						int offsetX = toX.y*P.size.x + toX.x;
						int offset_bwX = to_bwX.y*P.size.x + to_bwX.x;	//backwards offset is already negated!
						int offsetY = toY.y*P.size.x + toY.x;
						int offset_bwY = to_bwY.y*P.size.x + to_bwY.x;	//backwards offset is already negated!

						// check if offset leads to valid pixel
						bool valid1 = (f != nframes - 1) && (xx < P.size.x - to.x - 1)		&& (xx >= to.x)			&& (yy < P.size.y - to.y)		&& (yy >= to.y)		&& (xx != P.size.x - 1);
						bool valid2 = (f != nframes - 1) && (xx < P.size.x - toX.x)			&& (xx >= toX.x + 1)	&& (yy < P.size.y - toX.y)		&& (yy >= toX.y)	&& (xx != 0);
						bool valid3 = (f != 0)			 && (xx < P.size.x - to_bw.x - 1)	&& (xx >= to_bw.x)		&& (yy < P.size.y - to_bw.y)	&& (yy >= to_bw.y)	&& (xx != P.size.x - 1);
						bool valid4 = (f != 0)			 && (xx < P.size.x - to_bwX.x)		&& (xx >= to_bwX.x + 1) && (yy < P.size.y - to_bwX.y)	&& (yy >= to_bwX.y) && (xx != 0);

						if (valid4)	Axi +=  p_w2[fo + P.dxdt_idx*n + i - P.n_constr * n - 1 + offset_bwX]	* (xi - p_x[f*n + i - 1] - p_x[f*n + i - n + offset_bwX]	+ p_x[f*n + i - n - 1 + offset_bwX])		*w_dsdt2;
						if (valid3)	Axi +=  p_w2[fo + P.dxdt_idx*n + i - P.n_constr * n + offset_bw]		* (xi - p_x[f*n + i + 1] - p_x[f*n + i - n + offset_bw]		+ p_x[f*n + i - n + 1 + offset_bw])			*w_dsdt2;
						if (valid2)	Axi +=  p_w2[fo + P.dxdt_idx*n + i - 1]									* (xi - p_x[f*n + i - 1] - p_x[f*n + i + n + offsetX]		+ p_x[f*n + i + n - 1 + offsetX])			*w_dsdt2;
						if (valid1)	Axi +=  p_w2[fo + P.dxdt_idx*n + i]										* (xi - p_x[f*n + i + 1] - p_x[f*n + i + n + offset]		+ p_x[f*n + i + n + 1 + offset])			*w_dsdt2;

						valid1 = (f != nframes - 1) && (xx < P.size.x - to.x)		&& (xx >= to.x)		&& (yy < P.size.y - to.y - 1)		&& (yy >= to.y)			&& (yy != P.size.y - 1);
						valid2 = (f != nframes - 1) && (xx < P.size.x - toY.x)		&& (xx >= toY.x)	&& (yy < P.size.y - toY.y)			&& (yy >= toY.y + 1)	&& (yy != 0);
						valid3 = (f != 0)			&& (xx < P.size.x - to_bw.x)	&& (xx >= to_bw.x)	&& (yy < P.size.y - to_bw.y - 1)	&& (yy >= to_bw.y)		&& (yy != P.size.y - 1);
						valid4 = (f != 0)			&& (xx < P.size.x - to_bwY.x)	&& (xx >= to_bwY.x) && (yy < P.size.y - to_bwY.y)		&& (yy >= to_bwY.y + 1) && (yy != 0);

						if (valid4)	Axi +=  p_w2[fo + P.dydt_idx*n + i - P.n_constr * n - P.size.x + offset_bwY]	* (xi - p_x[f*n + i - P.size.x] - p_x[f*n + i - n + offset_bwY] + p_x[f*n + i - n - P.size.x + offset_bwY])	*w_dsdt2;
						if (valid3)	Axi +=  p_w2[fo + P.dydt_idx*n + i - P.n_constr * n + offset_bw]				* (xi - p_x[f*n + i + P.size.x] - p_x[f*n + i - n + offset_bw]	+ p_x[f*n + i - n + P.size.x + offset_bw])	*w_dsdt2;
						if (valid2)	Axi +=  p_w2[fo + P.dydt_idx*n + i - P.size.x]									* (xi - p_x[f*n + i - P.size.x] - p_x[f*n + i + n + offsetY]	+ p_x[f*n + i + n - P.size.x + offsetY])	*w_dsdt2;
						if (valid1)	Axi +=  p_w2[fo + P.dydt_idx*n + i]												* (xi - p_x[f*n + i + P.size.x] - p_x[f*n + i + n + offset]		+ p_x[f*n + i + n + P.size.x + offset])		*w_dsdt2;
					}

				}
				else{
					if (f != 0)           Axi += p_w2[fo + n * P.dt_idx + i - P.n_constr * n] * (xi - p_x[f*n + i - n]) *w_dt2;
					if (f != nframes - 1) Axi += p_w2[fo + n * P.dt_idx + i] * (xi - p_x[f*n + i + n]) *w_dt2;
					if (P.useDDxt){
						float w_dsdt2 = P.w_dsdt*P.w_dsdt;
						//this what is added to (A^t W^2 A)x when we add the dxdt and dydt constraints. Its quite a messy expression ;-)
						if (f != 0 && xx != 0)						Axi += p_w2[fo + P.dxdt_idx * n + i - P.n_constr * n - 1] * (xi - p_x[f*n + i - 1] - p_x[f*n + i - n] + p_x[f*n + i - n - 1])		* w_dsdt2;
						if (f != 0 && xx != P.size.x - 1)			Axi += p_w2[fo + P.dxdt_idx * n + i - P.n_constr * n] * (xi - p_x[f*n + i + 1] - p_x[f*n + i - n] + p_x[f*n + i - n + 1])		* w_dsdt2;
						if (f != nframes - 1 && xx != 0)			Axi += p_w2[fo + P.dxdt_idx * n + i - 1] * (xi - p_x[f*n + i - 1] - p_x[f*n + i + n] + p_x[f*n + i + n - 1])		* w_dsdt2;
						if (f != nframes - 1 && xx != P.size.x - 1)	Axi += p_w2[fo + P.dxdt_idx * n + i] * (xi - p_x[f*n + i + 1] - p_x[f*n + i + n] + p_x[f*n + i + n + 1])		* w_dsdt2;

						if (f != 0 && yy != 0)						Axi += p_w2[fo + P.dydt_idx * n + i - P.n_constr * n - P.size.x] * (xi - p_x[f*n + i - P.size.x] - p_x[f*n + i - n] + p_x[f*n + i - n - P.size.x]) * w_dsdt2;
						if (f != 0 && yy != P.size.y - 1)			Axi += p_w2[fo + P.dydt_idx * n + i - P.n_constr * n] * (xi - p_x[f*n + i + P.size.x] - p_x[f*n + i - n] + p_x[f*n + i - n + P.size.x]) * w_dsdt2;
						if (f != nframes - 1 && yy != 0)			Axi += p_w2[fo + P.dydt_idx * n + i - P.size.x] * (xi - p_x[f*n + i - P.size.x] - p_x[f*n + i + n] + p_x[f*n + i + n - P.size.x]) * w_dsdt2;
						if (f != nframes - 1 && yy != P.size.y - 1)	Axi += p_w2[fo + P.dydt_idx * n + i] * (xi - p_x[f*n + i + P.size.x] - p_x[f*n + i + n] + p_x[f*n + i + n + P.size.x]) * w_dsdt2;
					}
				}
			}

			p_Ax[f*n + i] = Axi;
			p_xAx[0] += xi * Axi;
		}
	}


	printf("%f\n", p_xAx[0].x);
    unmap(Ax,   (void*)p_Ax,    true);
    unmap(xAx,  (void*)p_xAx,   true);
    unmap(w2,   (void*)p_w2,    false);
    unmap(x,    (void*)p_x,     false);
}

//------------------------------------------------------------------------

void Backend::calc_axpy(Vector* axpy, Vec3f a, Vector* x, Vector* y)
{
    assert(axpy && axpy->numElems == x->numElems && axpy->bytesPerElem == sizeof(Vec3f));
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(y && y->numElems == x->numElems && y->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_axpy  = (Vec3f*)map(axpy);
    const Vec3f*    p_x     = (const Vec3f*)map(x);
    const Vec3f*    p_y     = (const Vec3f*)map(y);

    for (int i = 0; i < x->numElems; i++)
        p_axpy[i] = a * p_x[i] + p_y[i];

    unmap(axpy, (void*)p_axpy,  true);
    unmap(x,    (void*)p_x,     false);
    unmap(y,    (void*)p_y,     false);
}

//------------------------------------------------------------------------

void Backend::calc_xdoty(Vector* xdoty, Vector* x, Vector* y)
{
    assert(xdoty && xdoty->numElems == 1 && xdoty->bytesPerElem == sizeof(Vec3f));
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(y && y->numElems == x->numElems && y->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_xdoty = (Vec3f*)map(xdoty);
    const Vec3f*    p_x     = (const Vec3f*)map(x);
    const Vec3f*    p_y     = (const Vec3f*)map(y);

    p_xdoty[0] = 0.0f;
    for (int i = 0; i < x->numElems; i++)
        p_xdoty[0] += p_x[i] * p_y[i];

    unmap(xdoty,    (void*)p_xdoty, true);
    unmap(x,        (void*)p_x,     false);
    unmap(y,        (void*)p_y,     false);
}

//------------------------------------------------------------------------
// r -= Ap*(rz2/pAp)
// rz  = r'*r
void Backend::calc_r_rz(Vector* r, Vector* rz, Vector* Ap, Vector* rz2, Vector* pAp)
{
    assert(r && r->bytesPerElem == sizeof(Vec3f));
    assert(rz && rz->numElems == 1 && rz->bytesPerElem == sizeof(Vec3f));
    assert(Ap && Ap->numElems == r->numElems && Ap->bytesPerElem == sizeof(Vec3f));
    assert(rz2 && rz2->numElems == 1 && rz2->bytesPerElem == sizeof(Vec3f));
    assert(pAp && pAp->numElems == 1 && pAp->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_r     = (Vec3f*)map(r);
    Vec3f*          p_rz    = (Vec3f*)map(rz);
    const Vec3f*    p_Ap    = (const Vec3f*)map(Ap);
    const Vec3f*    p_rz2   = (const Vec3f*)map(rz2);
    const Vec3f*    p_pAp   = (const Vec3f*)map(pAp);

    Vec3f a = p_rz2[0] / max(p_pAp[0], Vec3f(FLT_MIN));
    p_rz[0] = 0.0f;
    for (int i = 0; i < r->numElems; i++)
    {
        Vec3f ri = p_r[i] - p_Ap[i] * a;
        p_r[i] = ri;
        p_rz[0] += ri * ri;
    }

    unmap(r,    (void*)p_r,     true);
    unmap(rz,   (void*)p_rz,    true);
    unmap(Ap,   (void*)p_Ap,    false);
    unmap(rz2,  (void*)p_rz2,   false);
    unmap(pAp,  (void*)p_pAp,   false);
}

//------------------------------------------------------------------------
// x += p*(rz2/pAp) 
// p = r + p*(rz/rz2)
void Backend::calc_x_p(Vector* x, Vector* p, Vector* r, Vector* rz, Vector* rz2, Vector* pAp)
{
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(p && p->numElems == x->numElems && p->bytesPerElem == sizeof(Vec3f));
    assert(r && r->numElems == x->numElems && r->bytesPerElem == sizeof(Vec3f));
    assert(rz && rz->numElems == 1 && rz->bytesPerElem == sizeof(Vec3f));
    assert(rz2 && rz2->numElems == 1 && rz2->bytesPerElem == sizeof(Vec3f));
    assert(pAp && pAp->numElems == 1 && pAp->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_x     = (Vec3f*)map(x);
    Vec3f*          p_p     = (Vec3f*)map(p);
    const Vec3f*    p_r     = (const Vec3f*)map(r);
    const Vec3f*    p_rz    = (const Vec3f*)map(rz);
    const Vec3f*    p_rz2   = (const Vec3f*)map(rz2);
    const Vec3f*    p_pAp   = (const Vec3f*)map(pAp);

    Vec3f a = p_rz2[0] / max(p_pAp[0], Vec3f(FLT_MIN));
    Vec3f b = p_rz[0] / max(p_rz2[0], Vec3f(FLT_MIN));
    for (int i = 0; i < x->numElems; i++)
    {
        Vec3f pi = p_p[i];
        p_x[i] += pi * a;
        p_p[i] = p_r[i] + pi * b;
    }

    unmap(x,    (void*)p_x,     true);
    unmap(p,    (void*)p_p,     true);
    unmap(r,    (void*)p_r,     false);
    unmap(rz,   (void*)p_rz,    false);
    unmap(rz2,  (void*)p_rz2,   false);
    unmap(pAp,  (void*)p_pAp,   false);
}

//------------------------------------------------------------------------

void Backend::calc_w2(Vector* w2, Vector* e, float reg)
{
    assert(w2 && w2->bytesPerElem == sizeof(float));
    assert(e && e->numElems == w2->numElems && e->bytesPerElem == sizeof(Vec3f));

    float*          p_w2    = (float*)map(w2);
    const Vec3f*    p_e     = (const Vec3f*)map(e);

    float w2sum = 0.0f;
    for (int i = 0; i < w2->numElems; i++)
    {
        float w2i = 1.0f / (length(p_e[i]) + reg);
        p_w2[i] = w2i;
        w2sum += w2i;
    }

    float coef = (float)w2->numElems / w2sum; // normalize so that average(w2) = 1
    for (int i = 0; i < w2->numElems; i++)
        p_w2[i] *= coef;

    unmap(w2,   (void*)p_w2,    true);
    unmap(e,    (void*)p_e,     false);
}

//------------------------------------------------------------------------
// Poisson preconditioner from the following paper:
//
// A Parallel Preconditioned Conjugate Gradient Solver for the Poisson Problem on a Multi-GPU Platform
// M. Ament, G. Knittel, D. Weiskopf, W. Straßer
// http://www.vis.uni-stuttgart.de/~amentmo/docs/ament-pcgip-PDP-2010.pdf
//
// Seems to make a minor difference with L2, but no difference at all with L1.

void Backend::calc_MIx(Vector* MIx, PoissonMatrix P, Vector* w2, Vector* x)
{
    assert(MIx && MIx->numElems == P.size.x * P.size.y && MIx->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
    assert(w2 && w2->numElems == P.size.x * P.size.y * 3 && w2->bytesPerElem == sizeof(float));
    assert(x && x->numElems == P.size.x * P.size.y && x->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_MIx       = (Vec3f*)map(MIx);
    const float*    p_w2        = (const float*)map(w2);
    const Vec3f*    p_x         = (const Vec3f*)map(x);
    int             n           = x->numElems;
    float           alphaSqr    = P.alpha * P.alpha;
    Vec3f*          t           = new Vec3f[n];
    Vec3f*          DIt         = new Vec3f[n];

    // t = K'*x
    // t = (I - inv(D)*L')*x
    // t = x - inv(D)*U*x
    // DIt = inv(D)*t

    for (int yy = 0, i = 0; yy < P.size.y; yy++)
    for (int xx = 0; xx < P.size.x; xx++, i++)
    {
        Vec3f Di = p_w2[n * 0 + i] * alphaSqr;
        Vec3f Uxi = 0.0f;
        if (xx != 0)            Di += p_w2[n * 1 + i - 1];
        if (xx != P.size.x - 1) Di += p_w2[n * 1 + i], Uxi -= p_w2[n * 1 + i] * p_x[i + 1];
        if (yy != 0)            Di += p_w2[n * 2 + i - P.size.x];
        if (yy != P.size.y - 1) Di += p_w2[n * 2 + i], Uxi -= p_w2[n * 2 + i] * p_x[i + P.size.x];
        t[i] = p_x[i] - Uxi / Di;
        DIt[i] = t[i] / Di;
    }

    // Mx = K*t
    // Mx = (I - L*inv(D))*t
    // Mx = t - L*DIt

    for (int yy = 0, i = 0; yy < P.size.y; yy++)
    for (int xx = 0; xx < P.size.x; xx++, i++)
    {
        Vec3f LDIti = 0.0f;
        if (xx != 0) LDIti -= p_w2[n * 1 + i - 1] * DIt[i - 1];
        if (yy != 0) LDIti -= p_w2[n * 2 + i - P.size.x] * DIt[i - P.size.x];
        p_MIx[i] = t[i] - LDIti;
    }

    unmap(MIx,  (void*)p_MIx,   true);
    unmap(w2,   (void*)p_w2,    false);
    unmap(x,    (void*)p_x,     false);
    delete[] t;
    delete[] DIt;
}

//------------------------------------------------------------------------

void Backend::tonemapSRGB(Vector* out, Vector* in, int idx, float scale, float bias)
{
    assert(out && out->bytesPerElem == sizeof(unsigned int));
    assert(in && in->numElems % out->numElems == 0 && in->bytesPerElem == sizeof(Vec3f));

    unsigned int*   p_out   = (unsigned int*)map(out);
    const Vec3f*    p_in    = (const Vec3f*)map(in);

    for (int i = 0; i < out->numElems; i++)
    {
        Vec3f color = p_in[i + idx * out->numElems];
        for (int c = 0; c < 3; c++)
        {
            float& t = color[c];
            t = t * scale + bias;
            t = (t <= 0.0031308f) ? 12.92f * t : 1.055f * powf(t, 1.0f / 2.4f) - 0.055f; // linear to sRGB
        }

        p_out[i] = 0xFF000000 |
            ((int)min(max(color.x * 255.0f + 0.5f, 0.0f), 255.0f) << 0) |
            ((int)min(max(color.y * 255.0f + 0.5f, 0.0f), 255.0f) << 8) |
            ((int)min(max(color.z * 255.0f + 0.5f, 0.0f), 255.0f) << 16);
    }

    unmap(out,  (void*)p_out,   true);
    unmap(in,   (void*)p_in,    false);
}

//------------------------------------------------------------------------

void Backend::tonemapLinear(Vector* out, Vector* in, int idx, float scaleMin, float scaleMax, bool hasNegative)
{
    assert(out && out->bytesPerElem == sizeof(unsigned int));
    assert(in && in->numElems % out->numElems == 0 && in->bytesPerElem % sizeof(float) == 0);
    int numComponents = (int)(in->bytesPerElem / sizeof(float));
    int totalComponents = out->numElems * numComponents;

    unsigned int*   p_out   = (unsigned int*)map(out);
    const float*    p_in    = (const float*)map(in);

    float inMin = +FLT_MAX;
    float inMax = -FLT_MAX;
    for (int i = 0; i < totalComponents; i++)
    {
        float t = p_in[i + idx * totalComponents];
        inMin = min(inMin, t);
        inMax = max(inMax, t);
    }

    float scale = min(max((hasNegative) ? 0.5f / max(max(-inMin, inMax), FLT_MIN) : 1.0f / max(inMax, FLT_MIN), scaleMin), scaleMax);
    float bias = (hasNegative) ? 0.5f : 0.0f;

    for (int i = 0; i < out->numElems; i++)
    {
        Vec3f color = 0.0f;
        for (int c = 0; c < 3; c++)
            color[c] = (c < numComponents) ? abs(p_in[i * numComponents + c + idx * totalComponents] * scale + bias) : color[c - 1];

        p_out[i] = 0xFF000000 |
            ((int)min(max(color.x * 255.0f + 0.5f, 0.0f), 255.0f) << 0) |
            ((int)min(max(color.y * 255.0f + 0.5f, 0.0f), 255.0f) << 8) |
            ((int)min(max(color.z * 255.0f + 0.5f, 0.0f), 255.0f) << 16);
    }

    unmap(out,  (void*)p_out,   true);
    unmap(in,   (void*)p_in,    false);
} 

//------------------------------------------------------------------------

Backend::Timer* Backend::allocTimer(void)
{
    Timer* timer = new Timer;
    timer->beginTicks = 0;
    return timer;
}

//------------------------------------------------------------------------

void Backend::freeTimer(Timer* timer)
{
    delete timer;
}

//------------------------------------------------------------------------

void Backend::beginTimer(Timer* timer)
{
    assert(timer);
    LARGE_INTEGER ticks;
    if (!QueryPerformanceCounter(&ticks))
    {
        printf("QueryPerformanceFrequency() failed!\n");
        exit(0);
    }
    timer->beginTicks = ticks.QuadPart;
}

//------------------------------------------------------------------------

float Backend::endTimer(Timer* timer)
{
    assert(timer);
    LARGE_INTEGER ticks;
    if (!QueryPerformanceCounter(&ticks))
    {
        printf("QueryPerformanceFrequency() failed!\n");
        exit(0);
    }
    return (float)((double)(ticks.QuadPart - timer->beginTicks) * m_timerTicksToSecs);
}

//------------------------------------------------------------------------
