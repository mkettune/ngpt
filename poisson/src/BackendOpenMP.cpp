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

#include "BackendOpenMP.hpp"
#include <omp.h>

using namespace poisson;

//------------------------------------------------------------------------

BackendOpenMP::BackendOpenMP(void)
{
}

//------------------------------------------------------------------------

BackendOpenMP::~BackendOpenMP(void)
{
}

//------------------------------------------------------------------------

void BackendOpenMP::set(Vector* x, float y)
{
    assert(x && x->bytesPerElem % sizeof(float) == 0);

    float*  p_x = (float*)x->ptr;
    int     n   = (int)(x->bytesTotal / sizeof(float));

#pragma omp parallel for
    for (int i = 0; i < n; i++)
        p_x[i] = y;
}

//------------------------------------------------------------------------

void BackendOpenMP::copy(Vector* x, Vector* y)
{
    assert(x && y && x->bytesTotal == y->bytesTotal);

    float*          p_x = (float*)x->ptr;
    const float*    p_y = (const float*)y->ptr;
    int             n = (int)(x->bytesTotal / sizeof(float));

#pragma omp parallel for
    for (int i = 0; i < n; i++)
        p_x[i] = p_y[i];
}


//=================== FOR COMMENTS SEE CUDA CODE =========================
// Compute P*x 
//
void BackendOpenMP::calc_Px(Vector* Px, PoissonMatrix P, Vector* x,		int nframes, Vector* mv)
{
	int nimg = P.n_constr;
	
	assert(Px && Px->numElems == P.size.x * P.size.y * nimg * nframes && Px->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
    assert(x && x->numElems == P.size.x * P.size.y *nframes && x->bytesPerElem == sizeof(Vec3f));

	const Vec3f*    p_mv;
	if (P.useMotion)
		p_mv     = (const Vec3f*)mv->ptr;

    Vec3f*          p_Px    = (Vec3f*)Px->ptr;
    const Vec3f*    p_x     = (const Vec3f*)x->ptr;
    int             n       = P.size.x * P.size.y;


#pragma omp parallel for
	for (int i = 0; i < n; i++){
		for (int f = 0; f < nframes; f++){
			int xx = i % P.size.x;
			int yy = i / P.size.x;
			
			int fo = nimg * f*n;
			Vec3f xi = p_x[f*n + i];
			p_Px[fo + n * 0 + i] = xi * P.alpha;
			p_Px[fo + n * 1 + i] = (xx != P.size.x - 1) ? p_x[f*n + i + 1] - xi : 0.0f;
			p_Px[fo + n * 2 + i] = (yy != P.size.y - 1) ? p_x[f*n + i + P.size.x] - xi : 0.0f;

			if(P.useTime){
				
				if (P.useMotion){
					Vec3f to = p_mv[f*n + i]; //correct
					int offset = to.y*P.size.x + to.x;
					bool valid = (xx < P.size.x - to.x) && (xx >= to.x) && (yy < P.size.y - to.y) && (yy >= to.y);	// check if offset leads to valid path

					p_Px[fo + n * P.dt_idx + i] = (f != nframes-1 && valid) ? P.w_dt * (p_x[f*n + i + n + offset] - xi) : 0.0f;

					if (P.useDDxt){
						bool valid_x = (f != nframes - 1) && (xx < P.size.x - to.x - 1) && (xx >= to.x) && (yy < P.size.y - to.y)		&& (yy >= to.y);
						bool valid_y = (f != nframes - 1) && (xx < P.size.x - to.x)		&& (xx >= to.x) && (yy < P.size.y - to.y - 1)	&& (yy >= to.y);

						p_Px[fo + n * P.dxdt_idx + i]	= valid_x ? P.w_dsdt*(p_x[f*n + i + n + 1 + offset] - p_x[f*n + i + n + offset] - p_x[f*n + i + 1] + xi) : 0.0f;
						p_Px[fo + n * P.dydt_idx + i]	= valid_y ? P.w_dsdt*(p_x[f*n + i + n + P.size.x + offset] - p_x[f*n + i + n + offset] - p_x[f*n + i + P.size.x] + xi) : 0.0f;
					}
				}
				else{
					p_Px[fo + n * P.dt_idx + i] = (f != nframes - 1) ? P.w_dt*(p_x[f*n + i + n] - xi) : 0.0f;

					if (P.useDDxt){
						p_Px[fo + n * P.dxdt_idx + i]	= (f != nframes - 1 && xx != P.size.x - 1) ? (p_x[f*n + i + n + 1]			- p_x[f*n + i + n]	- p_x[f*n + i + 1]			+ xi)*P.w_dsdt : 0.0f;
						p_Px[fo + n * P.dydt_idx + i]	= (f != nframes - 1 && yy != P.size.y - 1) ? (p_x[f*n + i + n + P.size.x]	- p_x[f*n + i + n]	- p_x[f*n + i + P.size.x]	+ xi)*P.w_dsdt : 0.0f;

					}
				}
			}
		}
	}
}

//=================== FOR COMMENTS SEE CUDA CODE =========================
// Compute (P'*W^2)*x. Note that P is transposed. 
//
void BackendOpenMP::calc_PTW2x(Vector* PTW2x, PoissonMatrix P, Vector* w2, Vector* x, int nframes, Vector* mv, Vector* mv_bw)
{
	int nimg = P.n_constr;

    assert(PTW2x && PTW2x->numElems == P.size.x * P.size.y * nframes && PTW2x->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);	
	assert(w2 && w2->numElems == P.size.x * P.size.y * nimg * nframes && w2->bytesPerElem == sizeof(float));
	assert(x && x->numElems == P.size.x * P.size.y * nimg * nframes && x->bytesPerElem == sizeof(Vec3f));

	const Vec3f* p_mv;
	const Vec3f* p_mv_bw;
	if (P.useMotion){
		p_mv     = (const Vec3f*)mv->ptr;
		p_mv_bw  = (const Vec3f*)mv_bw->ptr;
	}

    Vec3f*          p_PTW2x = (Vec3f*)PTW2x->ptr;
    const float*    p_w2    = (const float*)w2->ptr;
    const Vec3f*    p_x     = (const Vec3f*)x->ptr;
    int             n       = P.size.x * P.size.y;

#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			int xx = i % P.size.x;
			int yy = i / P.size.x;
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
							Vec3f toX	 = (xx != 0) ?			 p_mv[f*n + i - 1] : Vec3f(0);
							Vec3f to_bwX = (f != 0 && xx != 0) ? p_mv_bw[f*n + i - 1 - n] : Vec3f(0);
							Vec3f toY	 = (yy != 0) ?			 p_mv[f*n + i - P.size.x] : Vec3f(0);
							Vec3f to_bwY = (f != 0 && yy != 0) ? p_mv_bw[f*n + i - P.size.x - n] : Vec3f(0);


							int offset_bwX = to_bwX.y*P.size.x + to_bwX.x;	//backwards offset is already negated!
							int offset_bwY = to_bwY.y*P.size.x + to_bwY.x;	//backwards offset is already negated!

							bool valid1 = (f != nframes - 1) && (xx < P.size.x - to.x - 1)	  && (xx >= to.x)		  && (yy < P.size.y - to.y)     && (yy >= to.y);
							bool valid2 = (f != nframes - 1) && (xx < P.size.x - toX.x)		  && (xx >= toX.x + 1)	  && (yy < P.size.y - toX.y)    && (yy >= toX.y);
							bool valid3 = (f != 0)		     && (xx < P.size.x - to_bw.x - 1) && (xx >= to_bw.x)	  && (yy < P.size.y - to_bw.y)  && (yy >= to_bw.y);
							bool valid4 = (f != 0)			 && (xx < P.size.x - to_bwX.x)	  && (xx >= to_bwX.x + 1) && (yy < P.size.y - to_bwX.y) && (yy >= to_bwX.y);

							if (valid1)	PTW2xi +=  p_w2[fo + P.dxdt_idx*n + i]									 * p_x[fo + P.dxdt_idx*n + i]									* P.w_dsdt;
							if (valid2)	PTW2xi -=  p_w2[fo + P.dxdt_idx*n + i - 1]								 * p_x[fo + P.dxdt_idx*n + i - 1]								* P.w_dsdt;
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
						if (f != nframes - 1)	PTW2xi  -= p_w2[fo + P.dt_idx * n + i]					* p_x[fo + P.dt_idx * n + i]				* P.w_dt;
						if (f != 0)				PTW2xi  += p_w2[fo + P.dt_idx * n + i - P.n_constr * n]	* p_x[fo + P.dt_idx * n + i - P.n_constr * n] * P.w_dt;

						if (P.useDDxt){
							//A^t W^2 x (x,y,f) = all stuff from above PLUS
							if (f != nframes - 1 && xx != P.size.x - 1) PTW2xi +=  p_w2[fo + P.dxdt_idx * n + i]								* p_x[fo + P.dxdt_idx * n + i]								* P.w_dsdt;
							if (f != nframes - 1 && xx != 0)			PTW2xi -=  p_w2[fo + P.dxdt_idx * n + i - 1]							* p_x[fo + P.dxdt_idx * n + i - 1]							* P.w_dsdt;
							if (f != 0 && xx != P.size.x - 1)			PTW2xi -=  p_w2[fo + P.dxdt_idx * n + i - P.n_constr * n]				* p_x[fo + P.dxdt_idx * n + i - P.n_constr * n]				* P.w_dsdt;
							if (f != 0 && xx != 0)						PTW2xi +=  p_w2[fo + P.dxdt_idx * n + i - 1 - P.n_constr * n]			* p_x[fo + P.dxdt_idx * n + i - 1 - P.n_constr * n]			* P.w_dsdt;

							if (f != nframes - 1 && yy != P.size.y - 1) PTW2xi +=  p_w2[fo + P.dydt_idx * n + i]								* p_x[fo + P.dydt_idx * n + i]								* P.w_dsdt;
							if (f != nframes - 1 && yy != 0)			PTW2xi -=  p_w2[fo + P.dydt_idx * n + i - P.size.x]						* p_x[fo + P.dydt_idx * n + i - P.size.x]					* P.w_dsdt;
							if (f != 0 && yy != P.size.y - 1)			PTW2xi -=  p_w2[fo + P.dydt_idx * n + i - P.n_constr * n]				* p_x[fo + P.dydt_idx * n + i - P.n_constr * n]				* P.w_dsdt;
							if (f != 0 && yy != 0)						PTW2xi +=  p_w2[fo + P.dydt_idx * n + i - P.size.x - P.n_constr * n]	* p_x[fo + P.dydt_idx * n + i - P.size.x - P.n_constr * n]	* P.w_dsdt;
						}
					}
				}
				p_PTW2x[n*f + i] = PTW2xi;
			}
		}
}

//=================== FOR COMMENTS SEE CUDA CODE =========================
// Compute Ax = (P'*W^2*P)*x and xAx = x'*(P'*W^2*P)*x
// 
void BackendOpenMP::calc_Ax_xAx(Vector* Ax, Vector* xAx, PoissonMatrix P, Vector* w2, Vector* x, int nframes, Vector* mv, Vector* mv_bw)
{
	int nimg = P.n_constr;

    assert(Ax && Ax->numElems == P.size.x * P.size.y * nframes && Ax->bytesPerElem == sizeof(Vec3f));
    assert(xAx && xAx->numElems == 1 && xAx->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
	assert(w2 && w2->numElems == P.size.x * P.size.y * nimg * nframes && w2->bytesPerElem == sizeof(float));
    assert(x && x->numElems == P.size.x * P.size.y * nframes && x->bytesPerElem == sizeof(Vec3f));

	const Vec3f* p_mv;
	const Vec3f* p_mv_bw;
	if (P.useMotion){
		p_mv     = (const Vec3f*)mv->ptr;
		p_mv_bw  = (const Vec3f*)mv_bw->ptr;
	}

    Vec3f*          p_Ax        = (Vec3f*)Ax->ptr;
    Vec3f*          p_xAx       = (Vec3f*)xAx->ptr;
    const float*    p_w2        = (const float*)w2->ptr;
    const Vec3f*    p_x         = (const Vec3f*)x->ptr;
    int             n           = P.size.x * P.size.y;
    float           alphaSqr    = P.alpha * P.alpha;

    float xAx_0 = 0.0f;
    float xAx_1 = 0.0f;
    float xAx_2 = 0.0f;


#pragma omp parallel for reduction(+ : xAx_0, xAx_1, xAx_2)
		for (int i = 0; i < n; i++){
			int xx = i % P.size.x;
			int yy = i / P.size.x;
			
			for (int f=0; f < nframes; f++){
				Vec3f xi = p_x[f*n+ i];

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
						bool valid = (xx < P.size.x - to.x) && (xx >= to.x) && (yy < P.size.y - to.y) && (yy >= to.y);
						bool valid_bw = (xx < P.size.x - to_bw.x) && (xx >= to_bw.x) && (yy < P.size.y - to_bw.y) && (yy >= to_bw.y);

						if (f != nframes - 1 && valid)	Axi +=  p_w2[fo + P.dt_idx * n + i]									* (xi - p_x[f*n + i + n + offset])	  * w_dt2;
						if (f != 0 && valid_bw)			Axi +=  p_w2[fo + P.dt_idx * n + i + offset_bw - P.n_constr * n]	* (xi - p_x[f*n + i - n + offset_bw]) * w_dt2;

						if (P.useDDxt){
							float w_dsdt2 = P.w_dsdt*P.w_dsdt;

							Vec3f toX	 = (xx != 0) ?			 p_mv[f*n + i - 1] : Vec3f(0);
							Vec3f to_bwX = (f != 0 && xx != 0) ? p_mv_bw[f*n + i - 1 - n] : Vec3f(0);
							Vec3f toY	 = (yy != 0) ?			 p_mv[f*n + i - P.size.x] : Vec3f(0);
							Vec3f to_bwY = (f != 0 && yy != 0) ? p_mv_bw[f*n + i - P.size.x - n] : Vec3f(0);

							// compute offset
							int offsetX = toX.y*P.size.x + toX.x;
							int offset_bwX = to_bwX.y*P.size.x + to_bwX.x;	//backwards offset is already negated!
							int offsetY = toY.y*P.size.x + toY.x;
							int offset_bwY = to_bwY.y*P.size.x + to_bwY.x;	//backwards offset is already negated!

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
						if (f != 0)           Axi += p_w2[fo + n * P.dt_idx + i - P.n_constr * n]	* (xi - p_x[f*n + i - n]) *w_dt2;
						if (f != nframes - 1) Axi += p_w2[fo + n * P.dt_idx + i]					* (xi - p_x[f*n + i + n]) *w_dt2;
						if (P.useDDxt){
							float w_dsdt2 = P.w_dsdt*P.w_dsdt;
							//this is added to (A^t W^2 A)x when we add the dxdt and dydt constraints.
							if (f != 0 && xx != 0)						Axi += p_w2[fo + P.dxdt_idx * n + i - P.n_constr * n - 1]			* (xi - p_x[f*n + i - 1]		- p_x[f*n + i - n] + p_x[f*n + i - n - 1])		* w_dsdt2;
							if (f != 0 && xx != P.size.x - 1)			Axi += p_w2[fo + P.dxdt_idx * n + i - P.n_constr * n]				* (xi - p_x[f*n + i + 1]		- p_x[f*n + i - n] + p_x[f*n + i - n + 1])		* w_dsdt2;
							if (f != nframes - 1 && xx != 0)			Axi += p_w2[fo + P.dxdt_idx * n + i - 1]							* (xi - p_x[f*n + i - 1]		- p_x[f*n + i + n] + p_x[f*n + i + n - 1])		* w_dsdt2;
							if (f != nframes - 1 && xx != P.size.x - 1)	Axi += p_w2[fo + P.dxdt_idx * n + i]								* (xi - p_x[f*n + i + 1]		- p_x[f*n + i + n] + p_x[f*n + i + n + 1])		* w_dsdt2;

							if (f != 0 && yy != 0)						Axi += p_w2[fo + P.dydt_idx * n + i - P.n_constr * n - P.size.x]	* (xi - p_x[f*n + i - P.size.x] - p_x[f*n + i - n] + p_x[f*n + i - n - P.size.x]) * w_dsdt2;
							if (f != 0 && yy != P.size.y -1)			Axi += p_w2[fo + P.dydt_idx * n + i - P.n_constr * n]				* (xi - p_x[f*n + i + P.size.x] - p_x[f*n + i - n] + p_x[f*n + i - n + P.size.x]) * w_dsdt2;
							if (f != nframes - 1 && yy != 0)			Axi += p_w2[fo + P.dydt_idx * n + i - P.size.x]						* (xi - p_x[f*n + i - P.size.x] - p_x[f*n + i + n] + p_x[f*n + i + n - P.size.x]) * w_dsdt2;
							if (f != nframes - 1 && yy != P.size.y - 1)	Axi += p_w2[fo + P.dydt_idx * n + i]								* (xi - p_x[f*n + i + P.size.x] - p_x[f*n + i + n] + p_x[f*n + i + n + P.size.x]) * w_dsdt2;
						}
					}
				}

				p_Ax[f*n + i] = Axi;

				Vec3f xAxi = xi * Axi;
				xAx_0 += xAxi[0];
				xAx_1 += xAxi[1];
				xAx_2 += xAxi[2];
			}
		}

	printf("%f\n", xAx_0 + xAx_1 + xAx_2);

    p_xAx[0] = Vec3f(xAx_0, xAx_1, xAx_2);
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_axpy(Vector* axpy, Vec3f a, Vector* x, Vector* y)
{
    assert(axpy && axpy->numElems == x->numElems && axpy->bytesPerElem == sizeof(Vec3f));
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(y && y->numElems == x->numElems && y->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_axpy  = (Vec3f*)axpy->ptr;
    const Vec3f*    p_x     = (const Vec3f*)x->ptr;
    const Vec3f*    p_y     = (const Vec3f*)y->ptr;
    int             n       = x->numElems;

#pragma omp parallel for
    for (int i = 0; i < n; i++)
        p_axpy[i] = a * p_x[i] + p_y[i];
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_xdoty(Vector* xdoty, Vector* x, Vector* y)
{
    assert(xdoty && xdoty->numElems == 1 && xdoty->bytesPerElem == sizeof(Vec3f));
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(y && y->numElems == x->numElems && y->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_xdoty = (Vec3f*)xdoty->ptr;
    const Vec3f*    p_x     = (const Vec3f*)x->ptr;
    const Vec3f*    p_y     = (const Vec3f*)y->ptr;
    int             n       = x->numElems;

    float xdoty_0 = 0.0f;
    float xdoty_1 = 0.0f;
    float xdoty_2 = 0.0f;

#pragma omp parallel for reduction(+ : xdoty_0, xdoty_1, xdoty_2)
    for (int i = 0; i < n; i++)
    {
        Vec3f xdotyi = p_x[i] * p_y[i];
        xdoty_0 += xdotyi[0];
        xdoty_1 += xdotyi[1];
        xdoty_2 += xdotyi[2];
    }

    p_xdoty[0] = Vec3f(xdoty_0, xdoty_1, xdoty_2);
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_r_rz(Vector* r, Vector* rz, Vector* Ap, Vector* rz2, Vector* pAp)
{
    assert(r && r->bytesPerElem == sizeof(Vec3f));
    assert(rz && rz->numElems == 1 && rz->bytesPerElem == sizeof(Vec3f));
    assert(Ap && Ap->numElems == r->numElems && Ap->bytesPerElem == sizeof(Vec3f));
    assert(rz2 && rz2->numElems == 1 && rz2->bytesPerElem == sizeof(Vec3f));
    assert(pAp && pAp->numElems == 1 && pAp->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_r     = (Vec3f*)r->ptr;
    Vec3f*          p_rz    = (Vec3f*)rz->ptr;
    const Vec3f*    p_Ap    = (const Vec3f*)Ap->ptr;
    const Vec3f*    p_rz2   = (const Vec3f*)rz2->ptr;
    const Vec3f*    p_pAp   = (const Vec3f*)pAp->ptr;
    int             n       = r->numElems;

    Vec3f a = p_rz2[0] / max(p_pAp[0], Vec3f(FLT_MIN));
    float rz_0 = 0.0f;
    float rz_1 = 0.0f;
    float rz_2 = 0.0f;

#pragma omp parallel for reduction(+ : rz_0, rz_1, rz_2)
    for (int i = 0; i < n; i++)
    {
        Vec3f ri = p_r[i] - p_Ap[i] * a;
        p_r[i] = ri;

        Vec3f rzi = ri * ri;
        rz_0 += rzi[0];
        rz_1 += rzi[1];
        rz_2 += rzi[2];
    }

    p_rz[0] = Vec3f(rz_0, rz_1, rz_2);
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_x_p(Vector* x, Vector* p, Vector* r, Vector* rz, Vector* rz2, Vector* pAp)
{
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(p && p->numElems == x->numElems && p->bytesPerElem == sizeof(Vec3f));
    assert(r && r->numElems == x->numElems && r->bytesPerElem == sizeof(Vec3f));
    assert(rz && rz->numElems == 1 && rz->bytesPerElem == sizeof(Vec3f));
    assert(rz2 && rz2->numElems == 1 && rz2->bytesPerElem == sizeof(Vec3f));
    assert(pAp && pAp->numElems == 1 && pAp->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_x     = (Vec3f*)x->ptr;
    Vec3f*          p_p     = (Vec3f*)p->ptr;
    const Vec3f*    p_r     = (const Vec3f*)r->ptr;
    const Vec3f*    p_rz    = (const Vec3f*)rz->ptr;
    const Vec3f*    p_rz2   = (const Vec3f*)rz2->ptr;
    const Vec3f*    p_pAp   = (const Vec3f*)pAp->ptr;
    int             n       = x->numElems;

    Vec3f a = p_rz2[0] / max(p_pAp[0], Vec3f(FLT_MIN));
    Vec3f b = p_rz[0] / max(p_rz2[0], Vec3f(FLT_MIN));

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        Vec3f pi = p_p[i];
        p_x[i] += pi * a;
        p_p[i] = p_r[i] + pi * b;
    }
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_w2(Vector* w2, Vector* e, float reg)
{
    assert(w2 && w2->bytesPerElem == sizeof(float));
    assert(e && e->numElems == w2->numElems && e->bytesPerElem == sizeof(Vec3f));

    float*          p_w2    = (float*)w2->ptr;
    const Vec3f*    p_e     = (const Vec3f*)e->ptr;
    int             n       = w2->numElems;

    float w2sum = 0.0f;

#pragma omp parallel for reduction(+ : w2sum)
    for (int i = 0; i < n; i++)
    {
        float w2i = 1.0f / (length(p_e[i]) + reg);
        p_w2[i] = w2i;
        w2sum += w2i;
    }

    float coef = (float)n / w2sum; // normalize so that average(w2) = 1

#pragma omp parallel for
    for (int i = 0; i < n; i++)
        p_w2[i] *= coef;
}

//------------------------------------------------------------------------

void BackendOpenMP::tonemapSRGB(Vector* out, Vector* in, int idx, float scale, float bias)
{
    assert(out && out->bytesPerElem == sizeof(unsigned int));
    assert(in && in->numElems % out->numElems == 0 && in->bytesPerElem == sizeof(Vec3f));

    unsigned int*   p_out   = (unsigned int*)out->ptr;
    const Vec3f*    p_in    = (const Vec3f*)in->ptr;

#pragma omp parallel for
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
}

//------------------------------------------------------------------------
