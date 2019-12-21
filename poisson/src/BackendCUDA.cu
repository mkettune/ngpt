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

#include "BackendCUDA.hpp"
#include <stdio.h>

namespace poisson
{
//------------------------------------------------------------------------

#define globalThreadIdx (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y)))

#if __CUDA_ARCH__ < 350
template <class T> __device__ __forceinline__ T __ldg   (const T* in)   { return *in; }
#endif

template <class T> __device__ __forceinline__ T ld4     (const T& in)   { 
	T out; 
	for (int ofs = 0; ofs < sizeof(T); ofs += 4) 
		*(float*)((char*)&out + ofs) = __ldg((float*)((char*)&in + ofs)); return out; 
}

template <class T> __device__ __forceinline__ T ld8     (const T& in)   { T out; for (int ofs = 0; ofs < sizeof(T); ofs += 8) *(float2*)((char*)&out + ofs) = __ldg((float2*)((char*)&in + ofs)); return out; }
template <class T> __device__ __forceinline__ T ld16    (const T& in)   { T out; for (int ofs = 0; ofs < sizeof(T); ofs += 16) *(float4*)((char*)&out + ofs) = __ldg((float4*)((char*)&in + ofs)); return out; }

//------------------------------------------------------------------------

BackendCUDA::BackendCUDA(int device)
{
    assert(device >= -1);

    // No device specified => choose one.

    if (device == -1)
    {
        device = chooseDevice();
        if (device == -1)
            fail("No suitable CUDA device found!");
    }

    // Initialize.

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    cudaSetDevice(device);
    checkError();

    printf("Using CUDA device %d: %s\n", device, prop.name);
    if (prop.major < 3)
        fail("Compute capability 3.0 or higher is required!");

    m_maxGridWidth = prop.maxGridSize[0];
    m_blockDim = (prop.major >= 5) ? dim3(32, 2, 1) : dim3(32, 4, 1);
}

//------------------------------------------------------------------------

BackendCUDA::~BackendCUDA(void)
{
}

//------------------------------------------------------------------------

int BackendCUDA::chooseDevice(void)
{
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);

    int bestDevice = -1;
    cudaDeviceProp bestProp;
    memset(&bestProp, 0, sizeof(bestProp));

    for (int d = 0; d < numDevices; d++)
    {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);

        if (p.major < 3) // need at least sm_30
            continue;

        if (bestDevice == -1 ||
            (p.major != bestProp.major) ? (p.major > bestProp.major) :
            (p.minor != bestProp.minor) ? (p.minor > bestProp.minor) :
            (p.multiProcessorCount > bestProp.multiProcessorCount))
        {
            bestDevice = d;
            bestProp = p;
        }
    }

    checkError();
    return bestDevice;
}

//------------------------------------------------------------------------

void BackendCUDA::checkError(void)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        fail("CUDA runtime error: %s!", cudaGetErrorString(error));
}

//------------------------------------------------------------------------

BackendCUDA::Vector* BackendCUDA::allocVector(int numElems, size_t bytesPerElem)
{
    assert(numElems >= 0);
    assert(bytesPerElem > 0);

    Vector* x       = new Vector;
    x->numElems     = numElems;
    x->bytesPerElem = bytesPerElem;
    x->bytesTotal   = numElems * bytesPerElem;
    x->ptr          = NULL;

    cudaMalloc(&x->ptr, x->bytesTotal);
    checkError();
    return x;
}

//------------------------------------------------------------------------

void BackendCUDA::freeVector(Vector* x)
{
    if (x && x->ptr)
    {
        cudaFree(x->ptr);
        checkError();
    }
    delete x;
}

//------------------------------------------------------------------------

void* BackendCUDA::map(Vector* x)
{
    assert(x);
    void* ptr = NULL;
    cudaMallocHost(&ptr, x->bytesTotal);
    cudaMemcpy(ptr, x->ptr, x->bytesTotal, cudaMemcpyDeviceToHost);
    checkError();
    return ptr;
}

//------------------------------------------------------------------------

void BackendCUDA::unmap(Vector* x, void* ptr, bool modified)
{
    if (ptr)
    {
        if (modified && x)
            cudaMemcpy(x->ptr, ptr, x->bytesTotal, cudaMemcpyHostToDevice);
        cudaFreeHost(ptr);
        checkError();
    }
}

//------------------------------------------------------------------------

__global__ void kernel_set(float* x, float y, int numElems)
{
    int i = globalThreadIdx;
    if (i >= numElems)
        return;

    x[i] = y;
}

//------------------------------------------------------------------------

void BackendCUDA::set(Vector* x, float y)
{
    assert(x && x->bytesPerElem % sizeof(float) == 0);
    int numElems = (int)(x->bytesTotal / sizeof(float));

    cudaFuncSetCacheConfig(&kernel_set, cudaFuncCachePreferL1);
    kernel_set<<<gridDim(numElems), m_blockDim>>>
        ((float*)x->ptr, y, numElems);

    checkError();
}

//------------------------------------------------------------------------

void BackendCUDA::copy(Vector* x, Vector* y)
{
    assert(x && y && x->bytesTotal == y->bytesTotal);
    cudaMemcpy(x->ptr, y->ptr, x->bytesTotal, cudaMemcpyDeviceToDevice);
    checkError();
}

//------------------------------------------------------------------------

void BackendCUDA::read(void* ptr, Vector* x)
{
    assert(ptr && x);
    cudaMemcpy(ptr, x->ptr, x->bytesTotal, cudaMemcpyDeviceToHost);
    checkError();
}

//------------------------------------------------------------------------

void BackendCUDA::write(Vector* x, const void* ptr)
{
    assert(x && ptr);
    cudaMemcpy(x->ptr, ptr, x->bytesTotal, cudaMemcpyHostToDevice);
    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_Px(Vec3f* Px, const Vec3f* x, Vec2i size, 
	float w_tp, float w_ds, float w_dsds, float w_dsds2, float w_dt, float w_dsdt, 
	int nframes, const Vec3f* mv, int nconstr, 
	bool useTime, bool useMotion, bool useDDxt, bool useDDxx, bool useDDxy, int dxdy_idx, int dt_idx, int dxdt_idx, int dydt_idx)
{
    int xx = threadIdx.x + blockIdx.x * blockDim.x;
    int yy = threadIdx.y + blockIdx.y * blockDim.y;
    if (xx >= size.x || yy >= size.y)
        return;

    int i = xx + yy * size.x;
    int n = size.x * size.y;

	int offset, fo;
	bool valid, valid_x, valid_y;
	Vec3f xi, to;
	
	//add as many constraints as frames
	//each frame has constraints [I; dX; dY; ( dXdX; dYdY; dXdY; dT; dXdT; dYdT)] (the ones in brakets are optional)
	for (int f = 0; f < nframes; f++){
		fo = nconstr * f*n;
		xi = ld4(x[f*n + i]);

		//default constraints I, dx, dy
		Px[fo + n * 0 + i]				= xi * w_tp;
		Px[fo + n * 1 + i]				= (xx != size.x - 1) ? (ld4(x[f*n + i + 1]) - xi) * w_ds : 0.0f;
		Px[fo + n * 2 + i]				= (yy != size.y - 1) ? (ld4(x[f*n + i + size.x]) - xi) * w_ds : 0.0f;

		// time constraints
		if (useTime){
			
			// use time constraints with motion vectors (non-orthogonal time shifts)
			if (useMotion){
				to = ld4(mv[f*n + i]);																// read motion vector
				offset = to.y*size.x + to.x;															// compute offset
				valid = (xx < size.x - to.x) && (xx >= -to.x) && (yy < size.y - to.y) && (yy >= -to.y);	// check if offset leads to valid path

				// dt = (x+xo,y+yo,f+1) -(x,y,f)
				Px[fo + n * dt_idx + i]		= (f != nframes - 1 && valid) ? w_dt*(ld4(x[f*n + i + n + offset]) - xi) : 0.0f;

				if (useDDxt){

					valid_x = (f != nframes - 1) && (xx != size.x - 1) && (xx < size.x - to.x - 1) && (xx >= -to.x) && (yy < size.y - to.y)     && (yy >= -to.y);
					valid_y = (f != nframes - 1) && (yy != size.y - 1) && (xx < size.x - to.x)	   && (xx >= -to.x) && (yy < size.y - to.y - 1) && (yy >= -to.y);

					// ddxt = dx(x+xo,y+yo,f+1) - dx(x,y,f) = [(x+xo+1,y+yo,f+1) - (x+xo,y+yo,f+1)] - [(x+1,y,f) - (x,y,f)]
					// ddyt = dy(x+xo,y+yo,f+1) - dy(x,y,f) = [(x+xo,y+yo+1,f+1) - (x+xo,y+yo,f+1)] - [(x,y+1,f) - (x,y,f)]
					Px[fo + n * dxdt_idx + i]	= (valid_x) ? w_dsdt*(ld4(x[f*n + i + n + 1 + offset])		- ld4(x[f*n + i + n + offset]) - ld4(x[f*n + i + 1])	  + xi) : 0.0f;					
					Px[fo + n * dydt_idx + i]	= (valid_y) ? w_dsdt*(ld4(x[f*n + i + n + size.x + offset]) - ld4(x[f*n + i + n + offset]) - ld4(x[f*n + i + size.x]) + xi) : 0.0f;
				}
			}

			// use orthogonal time constraints
			else{
				// dt = (x,y,f+1) -(x,y,f)
				Px[fo + n * dt_idx + i]		= (f != nframes - 1) ?	w_dt*(ld4(x[f*n + i + n]) - xi) : 0.0f;

				if (useDDxt){
					// ddxt = dx(x,y,f+1) - dx(x,y,f) = [(x+1,y,f+1) - (x,y,f+1)] - [(x+1,y,f) - (x,y,f)]
					// ddyt = dy(x,y,f+1) - dy(x,y,f) = [(x,y+1,f+1) - (x,y,f+1)] - [(x,y+1,f) - (x,y,f)]
					Px[fo + n * dxdt_idx + i]	= (f != nframes - 1 && xx != size.x - 1) ? w_dsdt*(ld4(x[f*n + i + n + 1])		 - ld4(x[f*n + i + n])	- ld4(x[f*n + i + 1])	   + xi) : 0.0f;			
					Px[fo + n * dydt_idx + i]	= (f != nframes - 1 && yy != size.y - 1) ? w_dsdt*(ld4(x[f*n + i + n + size.x])  - ld4(x[f*n + i + n])  - ld4(x[f*n + i + size.x]) + xi) : 0.0f;
				}
			}
		}

	}
}

//------------------------------------------------------------------------

void BackendCUDA::calc_Px(Vector* Px, PoissonMatrix P, Vector* x, int nframes, Vector* mv)
{
	//int nconstr = P.useTime ? (P.useDDxt? 6 : 4) : 3;
	int nconstr = P.n_constr;

	assert(Px && Px->numElems == P.size.x * P.size.y * nconstr * nframes && Px->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
    assert(x && x->numElems == P.size.x * P.size.y * nframes && x->bytesPerElem == sizeof(Vec3f));

    cudaFuncSetCacheConfig(&kernel_Px, cudaFuncCachePreferL1);

    kernel_Px<<<gridDim(P.size.x, P.size.y), m_blockDim>>>
		((Vec3f*)Px->ptr, (const Vec3f*)x->ptr, P.size,
		P.w_tp, P.w_ds, P.w_dsds, P.w_dsds2, P.w_dt, P.w_dsdt,
		/* P.alpha, 1.0, 1.0,/*P.w_dt, P.w_dsdt,*/ nframes,
		mv != NULL ? (const Vec3f*)mv->ptr : NULL, 
		nconstr, P.useTime, P.useMotion, P.useDDxt, P.useDDxx, P.useDDxy, P.dxdy_idx, P.dt_idx, P.dxdt_idx, P.dydt_idx);

    checkError();
}

//------------------------------------------------------------------------


// We compute P^T * w^2 * x. Note that P is transposed.
__global__ void kernel_PTW2x(Vec3f* PTW2x, const float* w2, const Vec3f* x, Vec2i size, 
	float w_tp, float w_ds, float w_dsds, float w_dsds2, float w_dt, float w_dsdt, 
	int nframes, const Vec3f* mv, const Vec3f* mv_bw, int nconstr, 
	bool useTime, bool useMotion, bool useDDxt, bool useDDxx, bool useDDxy, int dxdy_idx, int dt_idx, int dxdt_idx, int dydt_idx)
{
    int xx = threadIdx.x + blockIdx.x * blockDim.x;
    int yy = threadIdx.y + blockIdx.y * blockDim.y;
    if (xx >= size.x || yy >= size.y)
        return;

    int i = xx + yy * size.x;
    int n = size.x * size.y;

	int fo, offset_bw, offset_bwX, offset_bwY;
	Vec3f PTW2xi, to, toX, toY, to_bw, to_bwX, to_bwY;

	for (int f=0; f < nframes; f++){
		fo = nconstr*n*f;
		
		PTW2xi							=  ld4(w2[fo + 0*n + i])		  * ld4(x[fo + 0*n + i]) * w_tp;

		if (xx != 0)            PTW2xi +=  ld4(w2[fo + 1*n + i - 1])      * ld4(x[fo + 1*n + i - 1])	*w_ds;
		if (xx != size.x - 1)   PTW2xi -=  ld4(w2[fo + 1*n + i])          * ld4(x[fo + 1*n + i])		*w_ds;

		if (yy != 0)            PTW2xi +=  ld4(w2[fo + 2*n + i - size.x]) * ld4(x[fo + 2*n + i - size.x]) *w_ds;
		if (yy != size.y - 1)   PTW2xi -=  ld4(w2[fo + 2*n + i])          * ld4(x[fo + 2*n + i])		  *w_ds;

		if (useTime){

			//time constriants with motion vectors.
			if (useMotion){
				// read backwards and forward motion vector
				to		=			 ld4(mv[f*n + i]);			
				to_bw	= (f != 0) ? ld4(mv_bw[f*n + i - n]) : Vec3f(0);

				// compute offset
			//	int offset = to.y*size.x + to.x;															
				offset_bw = to_bw.y*size.x + to_bw.x;	//backwards offset is already negated!

				if ((f != 0) && (xx < size.x - to_bw.x) && (xx >= -to_bw.x) && (yy < size.y - to_bw.y) && (yy >= -to_bw.y)){
					PTW2xi += ld4(w2[fo + dt_idx * n + i + offset_bw - nconstr*n]) 
								* ld4(x[fo + dt_idx * n + i + offset_bw - nconstr*n])	* w_dt;
				}

				if ((f != nframes - 1) && (xx < size.x - to.x) && (xx >= -to.x) && (yy < size.y - to.y) && (yy >= -to.y)){
					PTW2xi -= ld4(w2[fo + dt_idx * n + i]) 
								* ld4(x[fo + dt_idx * n + i])	* w_dt;
				}

				

				if (useDDxt){

					toX		= (xx != 0) ?			 ld4(mv[f*n + i - 1]) : Vec3f(0);
					to_bwX	= (f != 0 && xx != 0) ?  ld4(mv_bw[f*n + i - 1 - n]) : Vec3f(0);
					toY		= (yy != 0) ?			 ld4(mv[f*n + i - size.x]) : Vec3f(0);
					to_bwY	= (f != 0 && yy != 0) ?  ld4(mv_bw[f*n + i - size.x - n]) : Vec3f(0);

					// compute offset
					offset_bwX = to_bwX.y*size.x + to_bwX.x;	//backwards offset is already negated!
					offset_bwY = to_bwY.y*size.x + to_bwY.x;	//backwards offset is already negated!
					
					//DxDt
					if ((f != nframes - 1) && (xx != size.x - 1) && (xx < size.x - to.x - 1) && (xx >= -to.x) && (yy < size.y - to.y) && (yy >= -to.y))
					{
						PTW2xi +=  ld4(w2[fo + dxdt_idx*n + i])									
									* ld4(x[fo + dxdt_idx*n + i]) * w_dsdt;
					}
					if ((f != nframes - 1) && (xx != 0) && (xx < size.x - toX.x) && (xx >= -toX.x + 1) && (yy < size.y - toX.y) && (yy >= -toX.y))
					{
						PTW2xi -=  ld4(w2[fo + dxdt_idx*n + i - 1])
							* ld4(x[fo + dxdt_idx*n + i - 1]) * w_dsdt;
					}
					if ((f != 0) && (xx != size.x - 1) && (xx < size.x - to_bw.x - 1) && (xx >= -to_bw.x) && (yy < size.y - to_bw.y) && (yy >= -to_bw.y))
					{
						PTW2xi -=  ld4(w2[fo + dxdt_idx*n + i - nconstr*n + offset_bw])
									* ld4(x[fo + dxdt_idx*n + i - nconstr*n + offset_bw]) * w_dsdt;
					}
					if ((f != 0) && (xx != 0) && (xx < size.x - to_bwX.x) && (xx >= -to_bwX.x + 1) && (yy < size.y - to_bwX.y) && (yy >= -to_bwX.y))
					{
						PTW2xi +=  ld4(w2[fo + dxdt_idx*n + i - 1 - nconstr*n + offset_bwX])
									* ld4(x[fo + dxdt_idx*n + i - 1 - nconstr*n + offset_bwX]) * w_dsdt;

					}

					//DyDt
					if ((f != nframes - 1) && (yy != size.y-1) && (xx < size.x - to.x) && (xx >= -to.x) && (yy < size.y - to.y - 1) && (yy >= -to.y))
					{
						PTW2xi +=  ld4(w2[fo + dydt_idx*n + i])
									* ld4(x[fo + dydt_idx*n + i]) * w_dsdt;
					}
					if ((f != nframes - 1) && (yy != 0) && (xx < size.x - toY.x) && (xx >= -toY.x) && (yy < size.y - toY.y) && (yy >= -toY.y + 1))
					{
						PTW2xi -=  ld4(w2[fo + dydt_idx*n + i - size.x])
									* ld4(x[fo + dydt_idx*n + i - size.x]) * w_dsdt;
					}
					if ((f != 0) && (yy != size.y-1) && (xx < size.x - to_bw.x) && (xx >= -to_bw.x) && (yy < size.y - to_bw.y - 1) && (yy >= -to_bw.y))
					{
						PTW2xi -=  ld4(w2[fo + dydt_idx*n + i - nconstr*n + offset_bw])
									* ld4(x[fo + dydt_idx*n + i - nconstr*n + offset_bw]) * w_dsdt;
					}
					if ((f != 0) && (yy != 0) && (xx < size.x - to_bwY.x) && (xx >= -to_bwY.x) && (yy < size.y - to_bwY.y) && (yy >= -to_bwY.y + 1))
					{
						PTW2xi +=  ld4(w2[fo + dydt_idx*n + i - size.x - nconstr*n + offset_bwY])
									* ld4(x[fo + dydt_idx*n + i - size.x - nconstr*n + offset_bwY])	* w_dsdt;
					}

				}
	
			}

			//orthogonal time constraints
			else{

				if (f != 0)					PTW2xi +=  ld4(w2[fo + dt_idx * n + i - nconstr*n])	* ld4(x[fo + dt_idx * n + i - nconstr*n])	* w_dt;
				if (f != nframes - 1)		PTW2xi -=  ld4(w2[fo + dt_idx * n + i])				* ld4(x[fo + dt_idx * n + i])				* w_dt;
				
				
				if (useDDxt)
				{
					//adding dxdt and dydt to the Eq. system adds to A^tW^2x(x,y) the following expressions
					// x(x,y,f)*w2(x,y,f) - x(x-1,y,f)*w2(x-1,y,f) - x(x,y,f-1)*w2(x,y,f-1) + x(x-1,y,f-1)*w2(x-1,y,f-1)
					if (f != nframes - 1 && xx != size.x - 1)		PTW2xi +=  ld4(w2[fo + dxdt_idx*n + i])						* ld4(x[fo + dxdt_idx*n + i])						* w_dsdt;
					if (f != nframes - 1 && xx != 0)				PTW2xi -=  ld4(w2[fo + dxdt_idx*n + i - 1])					* ld4(x[fo + dxdt_idx*n + i - 1])					* w_dsdt;
					if (f != 0 && xx != size.x - 1)					PTW2xi -=  ld4(w2[fo + dxdt_idx*n + i - nconstr * n])		* ld4(x[fo + dxdt_idx*n + i - nconstr*n])			* w_dsdt;
					if (f != 0 && xx != 0)							PTW2xi +=  ld4(w2[fo + dxdt_idx*n + i - 1 - nconstr * n])	* ld4(x[fo + dxdt_idx*n + i - 1 - nconstr*n])		* w_dsdt;

					// x(x,y,f)*w2(x,y,f) - x(x,y-1,f)*w2(x,y-1,f) - x(x,y,f-1)*w2(x,y,f-1) + x(x,y-1,f-1)*w2(x,y-1,f-1)
					if (f!=nframes-1 && yy!=size.y-1)				PTW2xi +=  ld4(w2[fo + dydt_idx*n + i])						* ld4(x[fo + dydt_idx*n + i])						* w_dsdt;
					if (f!=nframes-1 && yy!=0)						PTW2xi -=  ld4(w2[fo + dydt_idx*n + i - size.x])			* ld4(x[fo + dydt_idx*n + i - size.x])				* w_dsdt;
					if (f!=0 && yy!=size.y-1)						PTW2xi -=  ld4(w2[fo + dydt_idx*n + i - nconstr*n])			* ld4(x[fo + dydt_idx*n + i - nconstr*n])			* w_dsdt;
					if (f!=0 && yy!=0)								PTW2xi +=  ld4(w2[fo + dydt_idx*n + i - size.x - nconstr*n])* ld4(x[fo + dydt_idx*n + i - size.x - nconstr*n])	* w_dsdt;
				}
			}

		}
		
		PTW2x[n*f + i] = PTW2xi;
	}
}

//------------------------------------------------------------------------

void BackendCUDA::calc_PTW2x(Vector* PTW2x, PoissonMatrix P, Vector* w2, Vector* x, int nframes, Vector* mv, Vector* mv_bw)
{
	//int nconstr = P.useTime ? (P.useDDxt? 6 : 4) : 3;
	int nconstr = P.n_constr;

	assert(PTW2x && PTW2x->numElems == nframes * P.size.x * P.size.y && PTW2x->bytesPerElem == sizeof(Vec3f));
	assert(P.size.x >= 0 && P.size.y >= 0);
	assert(w2 && w2->numElems == P.size.x * P.size.y * nconstr * nframes && w2->bytesPerElem == sizeof(float));
	assert(x && x->numElems == P.size.x * P.size.y * nconstr * nframes && x->bytesPerElem == sizeof(Vec3f));


    cudaFuncSetCacheConfig(&kernel_PTW2x, cudaFuncCachePreferL1);
    kernel_PTW2x<<<gridDim(P.size.x, P.size.y), m_blockDim>>>
		((Vec3f*)PTW2x->ptr, (const float*)w2->ptr, (const Vec3f*)x->ptr, P.size, 
		P.w_tp, P.w_ds, P.w_dsds, P.w_dsds2, P.w_dt, P.w_dsdt,
		/*P.alpha, 1.0, 1.0,*/ nframes, 
		mv != NULL ? (const Vec3f*)mv->ptr : NULL, 
		mv_bw != NULL ? (const Vec3f*)mv_bw->ptr : NULL,
		nconstr, P.useTime, P.useMotion, P.useDDxt, P.useDDxx, P.useDDxy, P.dxdy_idx, P.dt_idx, P.dxdt_idx, P.dydt_idx);

    checkError();
}

//------------------------------------------------------------------------
// Ax = (P' diag(W^2) P)x yields expression below
__global__ void kernel_Ax_xAx(Vec3f* Ax, Vec3f* xAx, const float* w2, const Vec3f* x, Vec2i size, int elemsPerThread,
	/*float alphaSqr, float w_dt2, float w_dsdt2, */
	float w_tp, float w_ds, float w_dsds, float w_dsds2, float w_dt, float w_dsdt,
	int nframes, const Vec3f* mv, const Vec3f* mv_bw,
	int nconstr, bool useTime, bool useMotion, bool useDDxt, bool useDDxx, bool useDDxy, int dxdy_idx, int dt_idx, int dxdt_idx, int dydt_idx)
{
    int xxbegin = threadIdx.x + blockIdx.x * (elemsPerThread * 32);
    int xxend   = ::min(xxbegin + elemsPerThread * 32, size.x);
    int yy      = threadIdx.y + blockIdx.y * blockDim.y;
    if (yy >= size.y || xxbegin >= xxend)
        return;

    Vec3f sum = 0.0f;
    int n = size.x * size.y;

	int offset, offsetX, offsetY, offset_bw, offset_bwX, offset_bwY, fo;
	Vec3f to, toX, toY, to_bw, to_bwX, to_bwY, xi, Axi;

	for (int xx = xxbegin; xx < xxend; xx += 32){
		int i = xx + yy * size.x;
		for (int f=0; f < nframes; f++){

			fo = nconstr*n*f;
			xi = ld4(x[f*n + i]);
			Axi				     =  ld4(w2[fo + 0 * n + i])			* xi * w_tp;

			if (xx != 0)            Axi +=  ld4(w2[fo + 1 * n + i - 1])       * (xi - ld4(x[f*n + i - 1])) * w_ds;
			if (xx != size.x - 1)   Axi +=  ld4(w2[fo + 1 * n + i])           * (xi - ld4(x[f*n + i + 1])) * w_ds;

			if (yy != 0)            Axi +=  ld4(w2[fo + 2 * n + i - size.x])  * (xi - ld4(x[f*n + i - size.x])) * w_ds;
			if (yy != size.y - 1)   Axi +=  ld4(w2[fo + 2 * n + i])           * (xi - ld4(x[f*n + i + size.x])) * w_ds;

			if (useTime){
				if (useMotion){
					//get forward and backward motion vector
					to		=			 ld4(mv[f*n + i]);
					to_bw	= (f != 0) ? ld4(mv_bw[f*n + i - n]) : Vec3f(0);

					// compute offsets
					offset		= to.y*size.x + to.x;
					offset_bw	= to_bw.y*size.x + to_bw.x;		//backwards offset is already negated!

					if ((f != nframes - 1)	&& (xx < size.x - to.x)		&& (xx >= -to.x)	&& (yy < size.y - to.y)		&& (yy >= -to.y))	 Axi +=  ld4(w2[fo + dt_idx * n + i])							* (xi - ld4(x[f*n + i + n + offset]))	* w_dt;
					if ((f != 0)			&& (xx < size.x - to_bw.x)	&& (xx >= -to_bw.x)	&& (yy < size.y - to_bw.y)	&& (yy >= -to_bw.y)) Axi +=  ld4(w2[fo + dt_idx * n + i + offset_bw - nconstr * n])	* (xi - ld4(x[f*n + i - n + offset_bw])) * w_dt;

					if (useDDxt){
						toX		= (xx != 0) ?			ld4(mv[f*n + i - 1]) : Vec3f(0);
						to_bwX	= (f != 0 && xx != 0) ? ld4(mv_bw[f*n + i - 1 - n]) : Vec3f(0);
						toY		= (yy != 0) ?			ld4(mv[f*n + i - size.x]) : Vec3f(0);
						to_bwY	= (f != 0 && yy != 0) ?	ld4(mv_bw[f*n + i - size.x - n]) : Vec3f(0);

						// compute offset
						offsetX		= toX.y*size.x		+ toX.x;
						offset_bwX	= to_bwX.y*size.x	+ to_bwX.x;	//backwards offset is already negated!
						offsetY		= toY.y*size.x		+ toY.x;
						offset_bwY	= to_bwY.y*size.x	+ to_bwY.x;	//backwards offset is already negated!
						
						if ((f != 0)			&& (xx < size.x - to_bwX.x)		&& (xx >= -to_bwX.x + 1)&& (yy < size.y - to_bwX.y) && (yy >= -to_bwX.y)&& (xx != 0))			Axi +=  ld4(w2[fo + dxdt_idx*n + i - nconstr*n - 1 + offset_bwX])	* (xi - ld4(x[f*n + i - 1]) - ld4(x[f*n + i - n + offset_bwX])	+ ld4(x[f*n + i - n - 1 + offset_bwX]))	*w_dsdt;
						if ((f != 0)			&& (xx < size.x - to_bw.x - 1)	&& (xx >= -to_bw.x)		&& (yy < size.y - to_bw.y)	&& (yy >= -to_bw.y)	&& (xx != size.x - 1))	Axi +=  ld4(w2[fo + dxdt_idx*n + i - nconstr*n + offset_bw])		* (xi - ld4(x[f*n + i + 1]) - ld4(x[f*n + i - n + offset_bw])	+ ld4(x[f*n + i - n + 1 + offset_bw]))	*w_dsdt;
						if ((f != nframes - 1)	&& (xx < size.x - toX.x)		&& (xx >= -toX.x + 1)	&& (yy < size.y - toX.y)	&& (yy >= -toX.y)	&& (xx != 0))			Axi +=  ld4(w2[fo + dxdt_idx*n + i - 1])							* (xi - ld4(x[f*n + i - 1]) - ld4(x[f*n + i + n + offsetX])		+ ld4(x[f*n + i + n - 1 + offsetX]))	*w_dsdt;
						if ((f != nframes - 1)	&& (xx < size.x - to.x - 1)		&& (xx >= -to.x)		&& (yy < size.y - to.y)		&& (yy >= -to.y)	&& (xx != size.x - 1))	Axi +=  ld4(w2[fo + dxdt_idx*n + i])								* (xi - ld4(x[f*n + i + 1]) - ld4(x[f*n + i + n + offset])		+ ld4(x[f*n + i + n + 1 + offset]))		*w_dsdt;
						
						if ((f != 0)			&& (xx < size.x - to_bwY.x)		&& (xx >= -to_bwY.x)	&& (yy < size.y - to_bwY.y)		&& (yy >= -to_bwY.y + 1)&& (yy != 0))			Axi +=  ld4(w2[fo + dydt_idx*n + i - nconstr*n - size.x + offset_bwY])	* (xi - ld4(x[f*n + i - size.x]) - ld4(x[f*n + i - n + offset_bwY]) + ld4(x[f*n + i - n - size.x + offset_bwY]))*w_dsdt;
						if ((f != 0)			&& (xx < size.x - to_bw.x)		&& (xx >= -to_bw.x)		&& (yy < size.y - to_bw.y - 1)	&& (yy >= -to_bw.y)		&& (yy != size.y - 1))	Axi +=  ld4(w2[fo + dydt_idx*n + i - nconstr*n + offset_bw])			* (xi - ld4(x[f*n + i + size.x]) - ld4(x[f*n + i - n + offset_bw])	+ ld4(x[f*n + i - n + size.x + offset_bw]))	*w_dsdt;
						if ((f != nframes - 1)	&& (xx < size.x - toY.x)		&& (xx >= -toY.x)		&& (yy < size.y - toY.y)		&& (yy >= -toY.y + 1)	&& (yy != 0))			Axi +=  ld4(w2[fo + dydt_idx*n + i - size.x])							* (xi - ld4(x[f*n + i - size.x]) - ld4(x[f*n + i + n + offsetY])	+ ld4(x[f*n + i + n - size.x + offsetY]))	*w_dsdt;
						if ((f != nframes - 1)	&& (xx < size.x - to.x)			&& (xx >= -to.x)		&& (yy < size.y - to.y - 1)		&& (yy >= -to.y)		&& (yy != size.y - 1))	Axi +=  ld4(w2[fo + dydt_idx*n + i])									* (xi - ld4(x[f*n + i + size.x]) - ld4(x[f*n + i + n + offset])		+ ld4(x[f*n + i + n + size.x + offset]))	*w_dsdt;
					}
				}
				else{
					if (f != 0)					Axi +=  ld4(w2[fo + dt_idx * n + i - nconstr * n])	* (xi - ld4(x[f*n + i - n])) * w_dt;
					if (f != nframes - 1)		Axi +=  ld4(w2[fo + dt_idx * n + i])				* (xi - ld4(x[f*n + i + n])) * w_dt;

					if (useDDxt)
					{
						//this what is added to (A^t W^2 A)x when we add the dxdt and dydt constraints. Its quite a messy expression ;-)
						if (f != 0 && xx != 0)						Axi += ld4(w2[fo + dxdt_idx * n + i - nconstr * n - 1])		* (xi - ld4(x[f*n + i - 1])		 - ld4(x[f*n + i - n]) + ld4(x[f*n + i - n - 1]))		* w_dsdt;
						if (f != 0 && xx != size.x - 1)				Axi += ld4(w2[fo + dxdt_idx * n + i - nconstr * n])			* (xi - ld4(x[f*n + i + 1])		 - ld4(x[f*n + i - n]) + ld4(x[f*n + i - n + 1]))		* w_dsdt;
						if (f != nframes - 1 && xx != 0)			Axi += ld4(w2[fo + dxdt_idx * n + i - 1])					* (xi - ld4(x[f*n + i - 1])		 - ld4(x[f*n + i + n]) + ld4(x[f*n + i + n - 1]))		* w_dsdt;
						if (f != nframes - 1 && xx != size.x - 1)	Axi += ld4(w2[fo + dxdt_idx * n + i])						* (xi - ld4(x[f*n + i + 1])		 - ld4(x[f*n + i + n]) + ld4(x[f*n + i + n + 1]))		* w_dsdt;

						if (f != 0 && yy != 0)						Axi += ld4(w2[fo + dydt_idx * n + i - nconstr * n - size.x])* (xi - ld4(x[f*n + i - size.x]) - ld4(x[f*n + i - n]) + ld4(x[f*n + i - n - size.x]))	* w_dsdt;
						if (f != 0 && yy != size.y - 1)				Axi += ld4(w2[fo + dydt_idx * n + i - nconstr * n])			* (xi - ld4(x[f*n + i + size.x]) - ld4(x[f*n + i - n]) + ld4(x[f*n + i - n + size.x]))	* w_dsdt;
						if (f != nframes - 1 && yy != 0)			Axi += ld4(w2[fo + dydt_idx * n + i - size.x])				* (xi - ld4(x[f*n + i - size.x]) - ld4(x[f*n + i + n]) + ld4(x[f*n + i + n - size.x]))	* w_dsdt;
						if (f != nframes - 1 && yy != size.y - 1)	Axi += ld4(w2[fo + dydt_idx * n + i])						* (xi - ld4(x[f*n + i + size.x]) - ld4(x[f*n + i + n]) + ld4(x[f*n + i + n + size.x]))	* w_dsdt;
					}
				}
			}

			Ax[f*n + i] = Axi;
			sum += xi * Axi;
		}

	}


    for (int c = 0; c < 3; c++)
    {
        float t = sum[c];
        for (int i = 1; i < 32; i *= 2) t += __shfl_xor(t, i);
        if (threadIdx.x == 0) atomicAdd(&xAx->x + c, t);
    }
}

//------------------------------------------------------------------------

void BackendCUDA::calc_Ax_xAx(Vector* Ax, Vector* xAx, PoissonMatrix P, Vector* w2, Vector* x, int nframes, Vector* mv, Vector* mv_bw)
{
	//int nconstr = P.useTime ? (P.useDDxt? 6 : 4) : 3;
	int nconstr = P.n_constr;

    assert(Ax && Ax->numElems == nframes * P.size.x * P.size.y && Ax->bytesPerElem == sizeof(Vec3f));
    assert(xAx && xAx->numElems == 1 && xAx->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
	assert(w2 && w2->numElems == P.size.x * P.size.y * nconstr * nframes&& w2->bytesPerElem == sizeof(float));
    assert(x && x->numElems == nframes * P.size.x * P.size.y && x->bytesPerElem == sizeof(Vec3f));

    int elemsPerThread = 2;
    int totalThreadsX = (P.size.x - 1) / elemsPerThread + 1;
    set(xAx, 0.0f);

    cudaFuncSetCacheConfig(&kernel_Ax_xAx, cudaFuncCachePreferL1);
    kernel_Ax_xAx<<<gridDim(totalThreadsX, P.size.y), m_blockDim>>>
		((Vec3f*)Ax->ptr, (Vec3f*)xAx->ptr, (const float*)w2->ptr, (const Vec3f*)x->ptr, P.size, 
		elemsPerThread, 
		P.w_tp*P.w_tp, P.w_ds*P.w_ds, P.w_dsds*P.w_dsds, P.w_dsds2*P.w_dsds2, P.w_dt*P.w_dt, P.w_dsdt*P.w_dsdt,
		/*P.alpha*P.alpha, 1.0, 1.0,/*P.w_dt*P.w_dt, P.w_dsdt*P.w_dsdt,*/ 
		nframes, 
		mv != NULL ? (const Vec3f*)mv->ptr : NULL,
		mv_bw != NULL ? (const Vec3f*)mv_bw->ptr : NULL,
		nconstr, P.useTime, P.useMotion, P.useDDxt, P.useDDxx, P.useDDxy, P.dxdy_idx, P.dt_idx, P.dxdt_idx, P.dydt_idx);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_axpy(Vec3f* axpy, Vec3f a, const Vec3f* x, const Vec3f* y, int numElems)
{
    int i = globalThreadIdx;
    if (i >= numElems)
        return;

    axpy[i] = a * ld4(x[i]) + ld4(y[i]);
}

//------------------------------------------------------------------------

void BackendCUDA::calc_axpy(Vector* axpy, Vec3f a, Vector* x, Vector* y)
{
    assert(axpy && axpy->numElems == x->numElems && axpy->bytesPerElem == sizeof(Vec3f));
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(y && y->numElems == x->numElems && y->bytesPerElem == sizeof(Vec3f));

    cudaFuncSetCacheConfig(&kernel_axpy, cudaFuncCachePreferL1);
    kernel_axpy<<<gridDim(x->numElems), m_blockDim>>>
        ((Vec3f*)axpy->ptr, a, (const Vec3f*)x->ptr, (const Vec3f*)y->ptr, x->numElems);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_xdoty(Vec3f* xdoty, const Vec3f* x, const Vec3f* y, int numElems, int elemsPerThread)
{
    int begin = globalThreadIdx;
    begin = (begin & 31) + (begin & ~31) * elemsPerThread;
    int end = ::min(begin + elemsPerThread * 32, numElems);
    if (begin >= end)
        return;

    Vec3f sum = 0.0f;
    for (int i = begin; i < end; i += 32)
        sum += ld4(x[i]) * ld4(y[i]);

    for (int c = 0; c < 3; c++)
    {
        float t = sum[c];
        for (int i = 1; i < 32; i *= 2) t += __shfl_xor(t, i);
        if (threadIdx.x == 0) atomicAdd(&xdoty->x + c, t);
    }
}

//------------------------------------------------------------------------

void BackendCUDA::calc_xdoty(Vector* xdoty, Vector* x, Vector* y)
{
    assert(xdoty && xdoty->numElems == 1 && xdoty->bytesPerElem == sizeof(Vec3f));
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(y && y->numElems == x->numElems && y->bytesPerElem == sizeof(Vec3f));

    int elemsPerThread = 2;
    int totalThreads = (x->numElems - 1) / elemsPerThread + 1;
    set(xdoty, 0.0f);

    cudaFuncSetCacheConfig(&kernel_xdoty, cudaFuncCachePreferL1);
    kernel_xdoty<<<gridDim(totalThreads), m_blockDim>>>
        ((Vec3f*)xdoty->ptr, (const Vec3f*)x->ptr, (const Vec3f*)y->ptr, x->numElems, elemsPerThread);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_r_rz(Vec3f* r, Vec3f* rz, const Vec3f* Ap, const Vec3f* rz2, const Vec3f* pAp, int numPixels, int elemsPerThread)
{
    int begin = globalThreadIdx;
    begin = (begin & 31) + (begin & ~31) * elemsPerThread;
    int end = ::min(begin + elemsPerThread * 32, numPixels);
    if (begin >= end)
        return;

    Vec3f a = ld4(*rz2) / max(ld4(*pAp), FLT_MIN);
    Vec3f sum = 0.0f;

    for (int i = begin; i < end; i += 32)
    {
        Vec3f ri = ld4(r[i]) - ld4(Ap[i]) * a;
        r[i] = ri;
        sum += ri * ri;
    }

    for (int c = 0; c < 3; c++)
    {
        float t = sum[c];
        for (int i = 1; i < 32; i *= 2) t += __shfl_xor(t, i);
        if (threadIdx.x == 0) atomicAdd(&rz->x + c, t);
    }
}

//------------------------------------------------------------------------

void BackendCUDA::calc_r_rz(Vector* r, Vector* rz, Vector* Ap, Vector* rz2, Vector* pAp)
{
    assert(r && r->bytesPerElem == sizeof(Vec3f));
    assert(rz && rz->numElems == 1 && rz->bytesPerElem == sizeof(Vec3f));
    assert(Ap && Ap->numElems == r->numElems && Ap->bytesPerElem == sizeof(Vec3f));
    assert(rz2 && rz2->numElems == 1 && rz2->bytesPerElem == sizeof(Vec3f));
    assert(pAp && pAp->numElems == 1 && pAp->bytesPerElem == sizeof(Vec3f));

    int elemsPerThread = 2;
    int totalThreads = (r->numElems - 1) / elemsPerThread + 1;
    set(rz, 0.0f);

    cudaFuncSetCacheConfig(&kernel_r_rz, cudaFuncCachePreferL1);
    kernel_r_rz<<<gridDim(totalThreads), m_blockDim>>>
        ((Vec3f*)r->ptr, (Vec3f*)rz->ptr, (const Vec3f*)Ap->ptr, (const Vec3f*)rz2->ptr, (const Vec3f*)pAp->ptr, r->numElems, elemsPerThread);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_x_p(Vec3f* x, Vec3f* p, const Vec3f* r, const Vec3f* rz, const Vec3f* rz2, const Vec3f* pAp, int numElems)
{
    int i = globalThreadIdx;
    if (i >= numElems)
        return;

    Vec3f rzv = ld4(*rz);
    Vec3f rz2v = ld4(*rz2);
    Vec3f pApv = ld4(*pAp);
    Vec3f a = rz2v / max(pApv, FLT_MIN);
    Vec3f b = rzv / max(rz2v, FLT_MIN);

    Vec3f pi = ld4(p[i]);
    x[i] += pi * a;
    p[i] = ld4(r[i]) + pi * b;
}

//------------------------------------------------------------------------

void BackendCUDA::calc_x_p(Vector* x, Vector* p, Vector* r, Vector* rz, Vector* rz2, Vector* pAp)
{
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(p && p->numElems == x->numElems && p->bytesPerElem == sizeof(Vec3f));
    assert(r && r->numElems == x->numElems && r->bytesPerElem == sizeof(Vec3f));
    assert(rz && rz->numElems == 1 && rz->bytesPerElem == sizeof(Vec3f));
    assert(rz2 && rz2->numElems == 1 && rz2->bytesPerElem == sizeof(Vec3f));
    assert(pAp && pAp->numElems == 1 && pAp->bytesPerElem == sizeof(Vec3f));

    cudaFuncSetCacheConfig(&kernel_x_p, cudaFuncCachePreferL1);
    kernel_x_p<<<gridDim(x->numElems), m_blockDim>>>
        ((Vec3f*)x->ptr, (Vec3f*)p->ptr, (const Vec3f*)r->ptr, (const Vec3f*)rz->ptr, (const Vec3f*)rz2->ptr, (const Vec3f*)pAp->ptr, x->numElems);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_w2sum(float* w2sum, const Vec3f* e, float reg, int numElems, int elemsPerThread)
{
    int begin = globalThreadIdx;
    begin = (begin & 31) + (begin & ~31) * elemsPerThread;
    int end = ::min(begin + elemsPerThread * 32, numElems);
    if (begin >= end)
        return;

    float sum = 0.0f;
    for (int i = begin; i < end; i += 32)
        sum += 1.0f / (length(ld4(e[i])) + reg);

    for (int i = 1; i < 32; i *= 2) sum += __shfl_xor(sum, i);
    if (threadIdx.x == 0) atomicAdd(w2sum, sum);
}

//------------------------------------------------------------------------

__global__ void kernel_w2(float* w2, const Vec3f* e, float reg, int numElems, float coef)
{
    int i = globalThreadIdx;
    if (i >= numElems)
        return;

    w2[i] = coef / (length(ld4(e[i])) + reg);
}

//------------------------------------------------------------------------

void BackendCUDA::calc_w2(Vector* w2, Vector* e, float reg)
{
    assert(w2 && w2->bytesPerElem == sizeof(float));
    assert(e && e->numElems == w2->numElems && e->bytesPerElem == sizeof(Vec3f));

    int elemsPerThread = 2;
    int totalThreads = (w2->numElems - 1) / elemsPerThread + 1;
    cudaMemset(w2->ptr, 0, sizeof(float));

    cudaFuncSetCacheConfig(&kernel_w2sum, cudaFuncCachePreferL1);
    kernel_w2sum<<<gridDim(totalThreads), m_blockDim>>>
        ((float*)w2->ptr, (const Vec3f*)e->ptr, reg, w2->numElems, elemsPerThread);

    float w2sum = 0.0f;
    cudaMemcpy(&w2sum, w2->ptr, sizeof(float), cudaMemcpyDeviceToHost);
    float coef = (float)w2->numElems / w2sum; // normalize so that average(w2) = 1

    cudaFuncSetCacheConfig(&kernel_w2, cudaFuncCachePreferL1);
    kernel_w2<<<gridDim(w2->numElems), m_blockDim>>>
        ((float*)w2->ptr, (const Vec3f*)e->ptr, reg, w2->numElems, coef);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_tonemapSRGB(unsigned int* out, const Vec3f* in, int numPixels, float scale, float bias)
{
    int i = globalThreadIdx;
    if (i >= numPixels)
        return;

    Vec3f color = ld4(in[i]);
    for (int c = 0; c < 3; c++)
    {
        float& t = color[c];
        t = t * scale + bias;
        t = (t <= 0.0031308f) ? 12.92f * t : 1.055f * powf(t, 1.0f / 2.4f) - 0.055f; // linear to sRGB
    }

    out[i] = 0xFF000000 |
        ((int)min(max(color.x * 255.0f + 0.5f, 0.0f), 255.0f) << 0) |
        ((int)min(max(color.y * 255.0f + 0.5f, 0.0f), 255.0f) << 8) |
        ((int)min(max(color.z * 255.0f + 0.5f, 0.0f), 255.0f) << 16);
}

//------------------------------------------------------------------------

void BackendCUDA::tonemapSRGB(Vector* out, Vector* in, int idx, float scale, float bias)
{
    assert(out && out->bytesPerElem == sizeof(unsigned int));
    assert(in && in->numElems % out->numElems == 0 && in->bytesPerElem == sizeof(Vec3f));

    cudaFuncSetCacheConfig(&kernel_tonemapSRGB, cudaFuncCachePreferL1);
    kernel_tonemapSRGB<<<gridDim(out->numElems), m_blockDim>>>
        ((unsigned int*)out->ptr, (const Vec3f*)in->ptr + idx * out->bytesTotal, out->numElems, scale, bias);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_tonemapLinearA(unsigned int* minmaxU32, const float* in, int numPixels, int numComponents)
{
    int i = globalThreadIdx;
    if (i >= numPixels)
        return;

    float inMin = +FLT_MAX;
    float inMax = -FLT_MAX;
    for (int c = 0; c < numComponents; c++)
    {
        float t = ld4(in[i * numComponents + c]);
        inMin = fminf(inMin, t);
        inMax = fmaxf(inMax, t);
    }

    unsigned int minU32 = __float_as_int(inMin);
    unsigned int maxU32 = __float_as_int(inMax);
    atomicMin(&minmaxU32[0], minU32 ^ (((int)minU32 >> 31) | 0x80000000u));
    atomicMax(&minmaxU32[1], maxU32 ^ (((int)maxU32 >> 31) | 0x80000000u));
}

//------------------------------------------------------------------------

__global__ void kernel_tonemapLinearB(unsigned int* out, const float* in, int numPixels, int numComponents, float scale, float bias)
{
    int i = globalThreadIdx;
    if (i >= numPixels)
        return;

    Vec3f color;
    for (int c = 0; c < 3; c++)
        color[c] = (c < numComponents) ? fabsf(ld4(in[i * numComponents + c]) * scale + bias) : color[c - 1];

    out[i] = 0xFF000000 |
        ((int)min(max(color.x * 255.0f + 0.5f, 0.0f), 255.0f) << 0) |
        ((int)min(max(color.y * 255.0f + 0.5f, 0.0f), 255.0f) << 8) |
        ((int)min(max(color.z * 255.0f + 0.5f, 0.0f), 255.0f) << 16);
}

//------------------------------------------------------------------------

void BackendCUDA::tonemapLinear(Vector* out, Vector* in, int idx, float scaleMin, float scaleMax, bool hasNegative)
{
    assert(out && out->numElems >= 2 && out->bytesPerElem == sizeof(unsigned int));
    assert(in && in->numElems % out->numElems == 0 && in->bytesPerElem % sizeof(float) == 0);
    int numComponents = (int)(in->bytesPerElem / sizeof(float));

    Vec2i minmaxU32(~0u, 0u);
    cudaMemcpy(out->ptr, &minmaxU32, sizeof(Vec2i), cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(&kernel_tonemapLinearA, cudaFuncCachePreferL1);
    kernel_tonemapLinearA<<<gridDim(out->numElems), m_blockDim>>>
        ((unsigned int*)out->ptr, (const float*)in->ptr + idx * out->bytesTotal, out->numElems, numComponents);

    cudaMemcpy(&minmaxU32, out->ptr, sizeof(Vec2i), cudaMemcpyDeviceToHost);
    minmaxU32.x ^= ~((int)minmaxU32.x >> 31) | 0x80000000u;
    minmaxU32.y ^= ~((int)minmaxU32.y >> 31) | 0x80000000u;
    float inMin = *(float*)&minmaxU32.x;
    float inMax = *(float*)&minmaxU32.y;

    float scale = min(max((hasNegative) ? 0.5f / max(max(-inMin, inMax), FLT_MIN) : 1.0f / max(inMax, FLT_MIN), scaleMin), scaleMax);
    float bias = (hasNegative) ? 0.5f : 0.0f;

    cudaFuncSetCacheConfig(&kernel_tonemapLinearB, cudaFuncCachePreferL1);
    kernel_tonemapLinearB<<<gridDim(out->numElems), m_blockDim>>>
        ((unsigned int*)out->ptr, (const float*)in->ptr + idx * out->bytesTotal, out->numElems, numComponents, scale, bias);

    checkError();
}

//------------------------------------------------------------------------

Backend::Timer* BackendCUDA::allocTimer(void)
{
    TimerCUDA* timerCUDA = new TimerCUDA;
    timerCUDA->beginTicks = 0;
    timerCUDA->beginEvent = NULL;
    timerCUDA->endEvent = NULL;

    cudaEventCreate(&timerCUDA->beginEvent);
    cudaEventCreate(&timerCUDA->endEvent);
    checkError();
    return timerCUDA;
}

//------------------------------------------------------------------------

void BackendCUDA::freeTimer(Timer* timer)
{
    TimerCUDA* timerCUDA = (TimerCUDA*)timer;
    if (timerCUDA)
    {
        cudaEventDestroy(timerCUDA->beginEvent);
        cudaEventDestroy(timerCUDA->endEvent);
        checkError();
        delete timerCUDA;
    }
}

//------------------------------------------------------------------------

void BackendCUDA::beginTimer(Timer* timer)
{
    assert(timer);
    TimerCUDA* timerCUDA = (TimerCUDA*)timer;
    cudaDeviceSynchronize();
    cudaEventRecord(timerCUDA->beginEvent);
    checkError();
}

//------------------------------------------------------------------------

float BackendCUDA::endTimer(Timer* timer)
{
    assert(timer);
    TimerCUDA* timerCUDA = (TimerCUDA*)timer;
    cudaEventRecord(timerCUDA->endEvent);
    cudaDeviceSynchronize();
    float millis = 0.0f;
    cudaEventElapsedTime(&millis, timerCUDA->beginEvent, timerCUDA->endEvent);
    checkError();
    return millis * 1.0e-3f;
}

//------------------------------------------------------------------------

dim3 BackendCUDA::gridDim(int totalThreadsX, int totalThreadsY)
{
    dim3 gd;
    gd.x = (totalThreadsX - 1) / m_blockDim.x + 1;
    gd.y = (totalThreadsY - 1) / m_blockDim.y + 1;
    gd.z = 1;

    while ((int)gd.x > m_maxGridWidth)
    {
        gd.x /= 2;
        gd.y *= 2;
    }
    return  gd;
}

//------------------------------------------------------------------------
} // namespace poisson
