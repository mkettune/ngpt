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

#include "Solver.hpp"
#include "BackendCUDA.hpp"
#include "BackendOpenMP.hpp"
#include "ImagePfmIO.hpp"
#include "lodepng.h"
#include <stdio.h>
#include <stdarg.h>

using namespace poisson;

//------------------------------------------------------------------------

Solver::Params::Params(void)
{
    setDefaults();
}

//------------------------------------------------------------------------

void Solver::Params::setDefaults(void)
{
	g0PFM.resize(1);
	g1PFM.resize(1);
	g2PFM.resize(1);
	g3PFM.resize(1);
	dxPFM.resize(1);
	dyPFM.resize(1);
	ddxtPFM.resize(1);
	ddytPFM.resize(1);
	ddxxPFM.resize(1);
	ddyyPFM.resize(1);
	ddxyPFM.resize(1);
	throughputPFM.resize(1);
    directPFM.resize(1);
	finalPFM.resize(1);
	finalPNG.resize(1);

	g0PFM[0]			= "";
	g1PFM[0]			= "";
	g2PFM[0]			= "";
	g3PFM[0]			= "";

    dxPFM [0]          = "";
    dyPFM [0]          = "";
    throughputPFM[0]   = "";
    directPFM[0]       = "";
    referencePFM    = "";
    alpha           = 0.2f;
	w_ds			= 1.f;
	w_dt			= 1.f;
	w_dsdt          = 1.f;

    indirectPFM     = "";
    finalPFM[0]        = "";

	g0PNG			= "";
	g1PNG			= "";
	g2PNG			= "";
	g3PNG			= "";

    dxPNG           = "";
    dyPNG           = "";
    throughputPNG   = "";
    directPNG       = "";
    referencePNG    = "";
    indirectPNG     = "";
    finalPNG[0]        = "";
    brightness      = 1.0f;

    verbose         = false;
    display         = false;

	useTime         = false;
	useMotion       = false;
	useDDxt			= false;

	useDDxx			= false;
	useDDxy			= false;
	energy_weights	= false;

	L2				= false;

	//backend       = "Naive";
	//backend		= "OpenMP";
	backend		= "Auto";

    cudaDevice      = -1;

	nframes	= 1;
	iframe	= 0;

	logFunc = LogFunction([](const std::string& message) { printf(message.c_str()); });

    setConfigPreset("L1D");
}

//------------------------------------------------------------------------

bool Solver::Params::setConfigPreset(const char* preset)
{
    // Base config.

    irlsIterMax = 1;
    irlsRegInit = 0.0f;
    irlsRegIter = 0.0f;
    cgIterMax   = 1;
    cgIterCheck = 100;
    cgPrecond   = false;
    cgTolerance = 0.0f;

    // "L1D" = L1 default config
    // ~1s for 1280x720 on GTX980, L1 error lower than MATLAB reference.

    if (strcmp(preset, "L1D") == 0)
    {
        irlsIterMax = 20;
        irlsRegInit = 0.05f;
        irlsRegIter = 0.5f;
        cgIterMax   = 50;
        return true;
    }

    // "L1Q" = L1 high-quality config
    // ~50s for 1280x720 on GTX980, L1 error as low as possible.

    if (strcmp(preset, "L1Q") == 0)
    {
        irlsIterMax = 64;
        irlsRegInit = 1.0f;
        irlsRegIter = 0.7f;
        cgIterMax   = 1000;
        return true;
    }

    // "L1L" = L1 legacy config
    // ~89s for 1280x720 on GTX980, L1 error equal to MATLAB reference.

    if (strcmp(preset, "L1L") == 0)
    {
        irlsIterMax = 7;
        irlsRegInit = 1.0e-4f;
        irlsRegIter = 1.0e-1f;
        cgIterMax   = 20000;
        cgTolerance = 1.0e-20f;
        return true;
    }

    // "L2D" = L2 default config
    // ~0.1s for 1280x720 on GTX980, L2 error equal to MATLAB reference.

    if (strcmp(preset, "L2D") == 0)
    {
        irlsIterMax = 1;
        irlsRegInit = 0.0f;
        irlsRegIter = 0.0f;
        cgIterMax   = 50; ///was 50...
        return true;
    }

    // "L2Q" = L2 high-quality config
    // ~0.5s for 1280x720 on GTX980, L2 error as low as possible.

    if (strcmp(preset, "L2Q") == 0)
    {
        irlsIterMax = 1;
        irlsRegInit = 0.0f;
        irlsRegIter = 0.0f;
        cgIterMax   = 500;
        return true;
    }

    return false;
}

//------------------------------------------------------------------------

void Solver::Params::sanitize(void)
{
    alpha       = max(alpha, 0.0f);
	w_dt		= max(w_dt, 0.0f);
	w_dsdt      = max(w_dsdt, 0.0f);
    brightness  = max(brightness, 0.0f);
    irlsIterMax = max(irlsIterMax, 1);
    irlsRegInit = max(irlsRegInit, 0.0f);
    irlsRegIter = max(irlsRegIter, 0.0f);
    cgIterMax   = max(cgIterMax, 1);
    cgIterCheck = max(cgIterCheck, 1);
    cgTolerance = max(cgTolerance, 0.0f);
	nframes		= max(nframes, 1);
}

//------------------------------------------------------------------------

void Solver::Params::setLogFunction(LogFunction function)
{
	logFunc = function;
}

//------------------------------------------------------------------------

Solver::Solver(const Params& params)
:   m_params        (params),

    m_size          (-1, -1),
    m_dx            (NULL),
    m_dy            (NULL),
    m_throughput    (NULL),
    m_direct        (NULL),
    m_reference     (NULL),

    m_backend       (NULL),
    m_b             (NULL),
    m_e             (NULL),
    m_w2            (NULL),
    m_x             (NULL),
    m_r             (NULL),
    m_z             (NULL),
	m_tmpImg		(NULL),
    m_p             (NULL),
    m_Ap            (NULL),
    m_rr            (NULL),
    m_rz            (NULL),
    m_rz2           (NULL),
    m_pAp           (NULL),
    m_tonemapped    (NULL),
	m_tonemapped_small (NULL),
    m_timerTotal    (NULL),
    m_timerIter     (NULL)
{
    m_params.sanitize();

	m_dx.resize(m_params.nframes);
	m_dy.resize(m_params.nframes);
	m_dt.resize(m_params.nframes);
	m_ddxt.resize(m_params.nframes);
	m_ddyt.resize(m_params.nframes);
	m_ddxx.resize(m_params.nframes);
	m_ddyy.resize(m_params.nframes);
	m_ddxy.resize(m_params.nframes);

	if (m_params.useMotion){
		m_motion.resize(m_params.nframes);
		m_motion_bw.resize(m_params.nframes);
	}
	m_throughput.resize(m_params.nframes);
    m_direct.resize(m_params.nframes);
}

//------------------------------------------------------------------------

void Solver::importImages(void)
{
	for (int f=0; f < m_params.nframes; f++){

		importImage(&m_dx[f], m_params.g0PFM[f].c_str());
		importImage(&m_dy[f], m_params.g1PFM[f].c_str());

		if (m_params.useTime)
			importImage(&m_dt[f], m_params.g2PFM[f].c_str());

		if (m_params.useDDxt){
			importImage(&m_ddxt[f], m_params.ddxtPFM[f].c_str());
			importImage(&m_ddyt[f], m_params.ddytPFM[f].c_str());
		}

		if (m_params.useMotion){
			importImage(&m_motion[f], m_params.motionPFM[f].c_str());
			importImage(&m_motion_bw[f], m_params.motionBW_PFM[f].c_str());
		}

		importImage(&m_throughput[f], m_params.throughputPFM[f].c_str());
	}

}


//------------------------------------------------------------------------

Solver::~Solver(void)
{
    if (m_backend)
    {
        m_backend->freeVector(m_b);
        m_backend->freeVector(m_e);
        m_backend->freeVector(m_w2);
        m_backend->freeVector(m_x);
        m_backend->freeVector(m_r);
        m_backend->freeVector(m_z);
		m_backend->freeVector(m_tmpImg);
        m_backend->freeVector(m_p);
        m_backend->freeVector(m_Ap);
        m_backend->freeVector(m_rr);
        m_backend->freeVector(m_rz);
        m_backend->freeVector(m_rz2);
        m_backend->freeVector(m_pAp);
        m_backend->freeVector(m_tonemapped);
		m_backend->freeVector(m_tonemapped_small);
        m_backend->freeTimer(m_timerTotal);
        m_backend->freeTimer(m_timerIter);
        delete m_backend;
    }

	/*
    delete[] m_dx;
    delete[] m_dy;
    delete[] m_throughput;
    delete[] m_direct;
    delete[] m_reference;
	*/
}

//------------------------------------------------------------------------

void Solver::setupBackend(void)
{
    assert(m_dx[0] && m_dy[0]);
    assert(m_size.x > 0 && m_size.y > 0);

    // CUDA backend?
    if (!m_backend && (m_params.backend == "CUDA" || m_params.backend == "Auto"))
    {
        int device = m_params.cudaDevice;
        if (device < 0)
            device = BackendCUDA::chooseDevice();	
        if (m_params.backend == "CUDA" || device != -1)
            m_backend = new BackendCUDA(device);
    }
	
    // OpenMP backend?
    if (!m_backend && (m_params.backend == "OpenMP" || m_params.backend == "Auto"))
    {
        printf("Using OpenMP backend\n");
        m_backend = new BackendOpenMP;
    }

    // Naive backend?
    if (!m_backend && (m_params.backend == "Naive" || m_params.backend == "Auto"))
    {
        printf("Using naive CPU backend\n");
        m_backend = new Backend;
    }

    // Otherwise => error.
    if (!m_backend)
        fail("Invalid backend specified '%s'!", m_params.backend);

    // Allocate backend objects.	
    int n           = m_size.x * m_size.y;
	int fn			= m_params.nframes*n;

	//figure out how many constraint sets we have per frame (up to 7)
	int nconstr = 3 + (m_params.useTime ? 1 : 0) + (m_params.useDDxt ? 2 : 0);

	m_b = m_backend->allocVector(fn * nconstr, sizeof(Vec3f));
	m_e = m_backend->allocVector(fn * nconstr, sizeof(Vec3f));
	m_w2 = m_backend->allocVector(fn * nconstr, sizeof(float));
	
	if (m_params.useMotion){
		m_m   = m_backend->allocVector(fn, sizeof(Vec3f));
		m_mbw = m_backend->allocVector(fn, sizeof(Vec3f));
	}
	else
	{
		m_m = NULL;
		m_mbw = NULL;
	}

    m_x             = m_backend->allocVector(fn,     sizeof(Vec3f));
    m_r             = m_backend->allocVector(fn,     sizeof(Vec3f));
    m_z             = m_backend->allocVector(fn,     sizeof(Vec3f));
	m_tmpImg        = m_backend->allocVector(n,		 sizeof(Vec3f));
    m_p             = m_backend->allocVector(fn,     sizeof(Vec3f));
    m_Ap            = m_backend->allocVector(fn,     sizeof(Vec3f));
    m_rr            = m_backend->allocVector(1,     sizeof(Vec3f));
    m_rz            = m_backend->allocVector(1,     sizeof(Vec3f));
    m_rz2           = m_backend->allocVector(1,     sizeof(Vec3f));
    m_pAp           = m_backend->allocVector(1,     sizeof(Vec3f));
    m_tonemapped    = m_backend->allocVector(fn,     sizeof(unsigned int));
	m_tonemapped_small    = m_backend->allocVector(n, sizeof(unsigned int));
    m_timerTotal    = m_backend->allocTimer();
    m_timerIter     = m_backend->allocTimer();

	
	// compute index of constraint per frame for each type of constraint
	//order is [TP; dx; dy;; dt; dxdt; dydt] (only the first 3 constraints are mandatory)
	m_P.tp_idx	 = 0;
	m_P.dx_idx   = 1;
	m_P.dy_idx   = 2;
	m_P.dt_idx	 = 3 ;
	m_P.dxdt_idx = 3  + (m_params.useTime ? 1 : 0);
	m_P.dydt_idx = 3  + (m_params.useTime ? 1 : 0) + 1;
	m_P.n_constr = nconstr/*3 + (m_params.useTime ? 1 : 0) + (m_params.useDDxt ? 2 : 0)*/;

    // Initialize P.
    m_P.size.x  = m_size.x;
    m_P.size.y  = m_size.y;

	m_P.useTime   = m_params.useTime;
	m_P.useMotion = m_params.useMotion;
	m_P.useDDxt	  = m_params.useDDxt;

	{
		m_P.w_tp    = m_params.alpha;
		m_P.w_ds	=  m_params.w_ds;
		m_P.w_dt	= m_params.useTime ? m_params.w_dt : 0.f;
		m_P.w_dsdt	= m_params.useDDxt ? m_params.w_dsdt : 0.f;
	}

	printf("\nw_tp =\t\t%f \nw_ds =\t\t%f \nw_dt =\t\t%f \nw_dsdt =\t%f\n", m_P.w_tp, m_P.w_ds, m_P.w_dt, m_P.w_dsdt);


    // Initialize b = vertcat(throughput_rgb * alpha, dx_rgb, dy_rgb)
	Vec3f* p_b = (Vec3f*)m_backend->map(m_b);

	Vec3f* p_m = NULL; Vec3f* p_mbw = NULL;
	if (m_params.useMotion){
		p_m = (Vec3f*)m_backend->map(m_m);
		p_mbw = (Vec3f*)m_backend->map(m_mbw);
	}

	//put whole constraint data into vector b
	for (int i = 0; i < n; i++){
		for (int f=0; f < m_params.nframes; f++){

			//primary data
			p_b[nconstr*n*f + m_P.tp_idx * n + i] = (m_throughput[f]) ? m_throughput[f][i] * m_P.w_tp : 0.0f;

			//dx, dy
			p_b[nconstr*n*f + m_P.dx_idx * n + i] = m_dx[f][i] * m_P.w_ds;
			p_b[nconstr*n*f + m_P.dy_idx * n + i] = m_dy[f][i] * m_P.w_ds;

			//dt
			if (m_params.useTime && f < m_params.nframes - 1)
				//if motion vectors are used some constraints are deactivated (i.e. the ones with a #INF in the motion map are set to zero)
				if (m_params.useMotion && std::isinf(m_motion[f][i].x))
					p_b[nconstr*n*f + m_P.dt_idx * n + i] = Vec3f(0.f);
				else
					p_b[nconstr*n*f + m_P.dt_idx * n + i] = m_dt[f][i] * m_P.w_dt;

			//dxdt, dydt
			if (m_params.useDDxt && f < m_params.nframes - 1){
				if (m_params.useMotion && std::isinf(m_motion[f][i].x)){
					p_b[nconstr*n*f + m_P.dxdt_idx * n + i] = Vec3f(0.f);
					p_b[nconstr*n*f + m_P.dydt_idx * n + i] = Vec3f(0.f);
				}else{
					p_b[nconstr*n*f + m_P.dxdt_idx * n + i] = m_ddxt[f][i] * m_P.w_dsdt;
					p_b[nconstr*n*f + m_P.dydt_idx * n + i] = m_ddyt[f][i] * m_P.w_dsdt;
				}
			}

			//store motion vector if needed
			if (m_params.useMotion && f < m_params.nframes - 1){
				p_m[n*f + i] = m_motion[f][i];
				p_mbw[n*f + i] = m_motion_bw[f][i];
			}
		}
	}
    m_backend->unmap(m_b, (void*)p_b, true);
	
	if (m_params.useMotion){
		m_backend->unmap(m_m, (void*)p_m, true);
		m_backend->unmap(m_mbw, (void*)p_mbw, true);
	}



    // Initialize x = throughput_rgb.


	if (m_throughput[0]){
		std::vector<Vec3f> artp;
		artp.resize(n*m_params.nframes);
		for (int i=0; i < m_params.nframes; i++)
			memcpy(&artp[i*n], m_throughput[i], sizeof(Vec3f)*n); //linearize image sequence into 1D array

		m_backend->write(m_x, artp.data());
	}
    else
        m_backend->set(m_x, 0.0f);
}

//------------------------------------------------------------------------
// Solve the indirect light image by minimizing the L1 (or L2) error.
//
// x = min_x L1(b - P*x),
// where
//      x = Nx1 vector representing the solution (elements are RGB triplets)
//      N = total number of pixels
//      b = (N*3)x1 vector representing the concatenation of throughput*alpha, dx, and dy.
//      P = (N*3)xN Poisson matrix that computes the equivalent of b from x.
//      L1() = L1 norm.
//
// We use Iteratively Reweighted Least Squares method (IRLS) to convert
// the L1 optimization problem to a series of L2 optimization problems.
// http://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
//
// x = min_x L2(W*b - W*P*x),
// where
//      W = (N*3)x(N*3) diagonal weight matrix, adjusted after each iteration
//
// We rewrite the L2 optimization problem using the normal equations.
// http://en.wikipedia.org/wiki/Linear_least_squares_%28mathematics%29
//
// (W*P)'*W*P*x = (W*P)'*W*b
// (P'*W^2*P)*x = (P'*W^2*b)
// A*x = bb,
// where
//      A  = P'*W^2*P = P'*diag(w2)*P = NxN matrix
//      bb = P'*W^2*b = P'*diag(w2)*b = Nx1 vector
//      w2 = (N*3)x1 vector of squared weights
//
// We then solve the normal equations using the Conjugate Gradient method (CG)
// or the Preconditioned Conjugate Gradient method (PCG).
// http://en.wikipedia.org/wiki/Conjugate_gradient_method

void Solver::solveIndirect(void)
{
    assert(m_dx[0] && m_dy[0] && m_backend && m_x);
    assert(m_size.x > 0 && m_size.y > 0);
    m_backend->beginTimer(m_timerTotal);
    // Loop over IRLS iterations.

    for (int irlsIter = 0; irlsIter < m_params.irlsIterMax; irlsIter++)
    {
        // Calculate approximation error.
		//printf("\nIRLS iteration: %d of %d\n\n", irlsIter+1, m_params.irlsIterMax);
        m_backend->calc_Px(m_e, m_P, m_x, m_params.nframes, m_m);          // e = P*x
        m_backend->calc_axpy(m_e, -1.0f, m_e, m_b); // e = b - P*x

        // Adjust weights.

		if (irlsIter == 0){
			m_backend->set(m_w2, 1.0f);
		}
        else
        {
            float reg = m_params.irlsRegInit * powf(m_params.irlsRegIter, (float)(irlsIter - 1));
            m_backend->calc_w2(m_w2, m_e, reg);     // w2 = coef / (length(e) + reg)
        }

        // Initialize conjugate gradient.

        Backend::Vector* rz = m_rz;
        Backend::Vector* rz2 = m_rz2;
        m_backend->calc_PTW2x(m_r, m_P, m_w2, m_e, m_params.nframes, m_m, m_mbw); // r = P'*diag(w2)*(b - P*x)
        m_backend->calc_xdoty(rz, m_r, m_r);        // rz = r'*r
        m_backend->copy(m_p, m_r);                  // p  = r

        for (int cgIter = 0;; cgIter++)
        {
            // Display status every N iterations.
		//	printf("\nconj. gradient iteration: %d of %d\n\n", cgIter, m_params.cgIterMax);

            if (cgIter % m_params.cgIterCheck == 0 || cgIter == m_params.cgIterMax)
            {
                // Calculate weighted L2 error.
			//	printf("calculate error\n");
                float errL2W;
                if (!m_params.cgPrecond || cgIter == 0)
                {
					const Vec3f* p_rz = (const Vec3f*)m_backend->map(rz); 
                    errL2W = p_rz->x + p_rz->y + p_rz->z;
                    m_backend->unmap(rz, (void*)p_rz, false);
                }
                else
                {
                    m_backend->calc_xdoty(m_rr, m_r, m_r);
                    const Vec3f* p_rr = (const Vec3f*)m_backend->map(m_rr);
                    errL2W = p_rr->x + p_rr->y + p_rr->z;
                    m_backend->unmap(m_rr, (void*)p_rr, false);
                }

                // Print status.

                std::string status = sprintf("IRLS = %-3d/ %d, CG = %-4d/ %d, errL2W = %9.2e",
                    irlsIter, m_params.irlsIterMax, cgIter, m_params.cgIterMax, errL2W);

                if (m_params.verbose)
                    printf("%s", status.c_str());

                // Done?

                if (cgIter == m_params.cgIterMax || errL2W <= m_params.cgTolerance)
                {
                    if (m_params.verbose)
                        printf("\n");
                    break;
                }

                // Display current solution.

                if (m_params.display)
                {
                    m_backend->tonemapSRGB(m_tonemapped, m_x, 0, m_params.brightness, 0.0f);
                    //m_backend->tonemapLinear(m_tonemapped, m_w2, 0, 0.0f, FLT_MAX, false);
                    display(sprintf("Poisson solver: %s", status.c_str()).c_str());
                }

                // Begin iteration timer.

                if (m_params.verbose)
                    m_backend->beginTimer(m_timerIter);
            }

            // Regular conjugate gradient iteration.

            if (!m_params.cgPrecond)
            {
                swap(rz, rz2);   // rz2 = rz
                m_backend->calc_Ax_xAx(m_Ap, m_pAp, m_P, m_w2, m_p, m_params.nframes, m_m, m_mbw);		// Ap = A*p, pAp = p'*A*p
                m_backend->calc_r_rz(m_r, rz, m_Ap, rz2, m_pAp);								// r -= Ap*(rz2/pAp), rz  = r'*r
                m_backend->calc_x_p(m_x, m_p, m_r, rz, rz2, m_pAp);								// x += p*(rz2/pAp), p = r + p*(rz/rz2)

            }


            // Preconditioned conjugate gradient iteration.

            else
            {
                if (cgIter == 0)
                {
                    m_backend->calc_MIx(m_z, m_P, m_w2, m_r);           // z  = inv(M)*r
                    m_backend->calc_xdoty(rz, m_r, m_z);                // rz = r'*z
                    m_backend->copy(m_p, m_z);                          // p  = z
                }

                swap(rz, rz2);                                          // rz2 = rz
				m_backend->calc_Ax_xAx(m_Ap, m_pAp, m_P, m_w2, m_p, m_params.nframes, m_m);    // Ap = A*p, pAp = p'*A*p
                m_backend->calc_r_rz(m_r, rz, m_Ap, rz2, m_pAp);        // r -= Ap*(rz2/pAp)
                m_backend->calc_MIx(m_z, m_P, m_w2, m_r);               // z = inv(M)*r
                m_backend->calc_xdoty(rz, m_r, m_z);                    // rz = r'*z
                m_backend->calc_x_p(m_x, m_p, m_r, rz, rz2, m_pAp);     // x += p*(rz2/pAp), p = r + p*(rz/rz2)
            }

            // Print iteration time every N iterations.

            if (m_params.verbose && cgIter % m_params.cgIterCheck == 0)
                printf(", %-5.2fms/iter\n", m_backend->endTimer(m_timerIter) * 1.0e3f);
        }
    }

    // Print total execution time.

    printf("Execution time = %.2f s\n", m_backend->endTimer(m_timerTotal));

    // Display final result.

    if (m_params.display)
    {
        m_backend->tonemapSRGB(m_tonemapped, m_x, 0, m_params.brightness, 0.0f);
        display("Poisson solver: Done");
    }
}

//------------------------------------------------------------------------

void Solver::evaluateMetrics(void)
{
    assert(m_dx[0] && m_dy[0] && m_backend && m_x);
    assert(m_size.x > 0 && m_size.y > 0);
    int n = m_size.x * m_size.y;

    // Calculate L1 and L2 error.
    {
        m_backend->calc_Px(m_e, m_P, m_x, m_params.nframes);          // e = P*x
        m_backend->calc_axpy(m_e, -1.0f, m_e, m_b); // e = b - P*x
        const Vec3f* p_e = (const Vec3f*)m_backend->map(m_e);

        float errL1 = 0.0f;
        float errL2 = 0.0f;
        for (int i = 0; i < n * 3; i++)
        {
            errL1 += length(p_e[i]);
            errL2 += lenSqr(p_e[i]);
        }
        errL1 /= (float)(n * 3);
        errL2 /= (float)(n * 3);

        printf("L1 error = %g\n", errL1);
        printf("L2 error = %g\n", errL2);
        m_backend->unmap(m_e, (void*)p_e, false);
    }
}

//------------------------------------------------------------------------

void Solver::exportImages(void)
{
	// Final.
	{
		if (m_direct[0]) {
			int n = m_size.x * m_size.y;
			int fn = m_params.nframes*n;

			std::vector<Vec3f> data;
			data.resize(n*m_params.nframes);
			for (int i=0; i < m_params.nframes; i++)
				memcpy(&data[i*n], m_direct[i], sizeof(Vec3f)*n); //linearize image sequence into 1D array

			m_backend->write(m_r, data.data());
			m_backend->calc_axpy(m_r, 1.0f, m_r, m_x);
		}
		else
		{
			m_backend->copy(m_r, m_x);
		}

		const Vec3f* p_r = (const Vec3f*)m_backend->map(m_r);
		//	for (int i=0; i < m_params.nframes; i++)
		exportImage2(p_r, m_params.finalPFM, m_params.finalPNG, m_z); //!
		m_backend->unmap(m_r, (void*)p_r, false);
	}
}

//------------------------------------------------------------------------

void Solver::importImage(Vec3f** pixelPtr, const char* pfmPath)
{
    // No file specified => skip.

    if (!pfmPath || !pfmPath[0])
        return;

    // Import.

    assert(pixelPtr);
    Vec3f* pixels = NULL;
    int width = 0;
    int height = 0;

    if (m_params.verbose)
        printf("Importing '%s'...\n", pfmPath);
    const char* error = importPfmImage(&pixels, &width, &height, pfmPath);
    if (error)
        fail("Failed to import '%s': %s", pfmPath, error);

    // Set/check dimensions.

    if (m_size.x == -1 && m_size.y == -1)
        m_size = Vec2i(width, height);
    else if (width != m_size.x || height != m_size.y)
        fail("Mismatch in image dimensions for '%s'!", pfmPath);

    // Set pixels.

    *pixelPtr = pixels;
}

//------------------------------------------------------------------------

void Solver::setImages(const float * const h_ib, const float * const h_dx, const float * const h_dy, const int width, const int height)
{
    assert(h_ib && h_dx && h_dy && width > 0 && height > 0);
    m_size = Vec2i(width, height);

    auto size  = width * height;
    auto bytes = size * sizeof(Vec3f);
	for (int f=0; f < m_params.nframes; f++){
		m_dx[f]		 = new Vec3f[size];
		m_dy[f]		 = new Vec3f[size];
		m_throughput[f] = new Vec3f[size];
        m_direct[f] = new Vec3f[size];
		memcpy(m_dx[f], h_dx, bytes);
		memcpy(m_dy[f], h_dy, bytes);
		memcpy(m_throughput[f], h_ib, bytes);

		importImage(&m_dx[f], m_params.dxPFM[f].c_str());
		importImage(&m_dy[f], m_params.dyPFM[f].c_str());
		importImage(&m_throughput[f], m_params.throughputPFM[f].c_str());
        importImage(&m_direct[f], m_params.directPFM[f].c_str());
	}
}

//------------------------------------------------------------------------

void Solver::getImage(float *h_dst)
{
    assert(h_dst);
    m_backend->copy(m_r, m_x);

    const Vec3f* p_r = (const Vec3f*)m_backend->map(m_r);

    for (int i = 0; i < m_size.x * m_size.y; i++){
        h_dst[3 * i]     = p_r[i].x;
        h_dst[3 * i + 1] = p_r[i].y;
        h_dst[3 * i + 2] = p_r[i].z;
    }
    m_backend->unmap(m_r, (void*)p_r, false);
}

//------------------------------------------------------------------------

void Solver::exportImage(const Vec3f* pixelPtr, const char* pfmPath, const char* pngPath, bool gradient, Backend::Vector* tmp)
{
    assert(m_size.x > 0 && m_size.y > 0);
    assert(tmp && tmp->numElems == m_size.x * m_size.y && tmp->bytesPerElem == sizeof(Vec3f));

    // Export as PFM.

   // if (pixelPtr && pfmPath && pfmPath[0])
    //{
     //   if (m_params.verbose)
     //       printf("Exporting '%s'...\n", pfmPath);
       // const char* error = exportPfmImage(pfmPath, pixelPtr, m_size.x, m_size.y);
       // if (error)
        //    fail("Failed to export '%s': %s", pfmPath, error);
    //}

    // Export as PNG.

    if (pixelPtr && pngPath && pngPath[0])
    {
        m_backend->write(tmp, pixelPtr);
        m_backend->tonemapSRGB(m_tonemapped, tmp, 0, m_params.brightness, (gradient) ? 0.5f : 0.0f);
        const unsigned int* p_tonemapped = (const unsigned int*)m_backend->map(m_tonemapped);

        if (m_params.verbose)
            printf("Exporting '%s'...\n", pngPath);
        if (lodepng_encode32_file(pngPath, (const unsigned char*)p_tonemapped, m_size.x, m_size.y) != 0)
            printf("1 Failed to export '%s'!", pngPath);

        m_backend->unmap(m_tonemapped, (void*)p_tonemapped, false);
    }
}


void Solver::exportImage2(const Vec3f* pixelPtr, std::vector<std::string> pfmPath, std::vector<std::string> pngPath, Backend::Vector* tmp)
{
	assert(m_size.x > 0 && m_size.y > 0);
	assert(tmp && tmp->numElems == m_size.x * m_size.y * m_params.nframes && tmp->bytesPerElem == sizeof(Vec3f));

	// Export as PFM.

	for (int i=0; i < m_params.nframes;i++)
		if (pixelPtr && pfmPath[i].c_str())
		{
			//if (m_params.verbose)
			printf("Exporting '%s'...\n", pfmPath[i].c_str());
			if (exportPfmImage(pfmPath[i].c_str(), &pixelPtr[i*m_size.x*m_size.y], m_size.x, m_size.y))
				fail("3 Failed to export '%s': %s", pfmPath[i]);
		}
		
	// Export as PNG.

	if (pixelPtr/* && pngPath[0].c_str()*/)
	{
		m_backend->write(tmp, pixelPtr);
		m_backend->tonemapSRGB(m_tonemapped, tmp, 0, m_params.brightness, 0.0f);
		const unsigned int* p_tonemapped = (const unsigned int*)m_backend->map(m_tonemapped);

		for (int i=0; i < m_params.nframes; i++){
			if (pngPath[i]!=""){
				//if (m_params.verbose)
				printf("Exporting '%s'...\n", pngPath[i].c_str());
				if (lodepng_encode32_file(pngPath[i].c_str(), (const unsigned char*)&p_tonemapped[i*m_size.x*m_size.y], m_size.x, m_size.y) != 0)
					printf("4 Failed to export '%s'!", pngPath[i]);
			}
		}

		m_backend->unmap(m_tonemapped, (void*)p_tonemapped, false);
	}
}

//------------------------------------------------------------------------

/*Marco: make this mitsuba friendlier*/
void Solver::importImagesMTS(float *dx, float *dy, float *dt, float *tp, int width, int height, int frame){

	m_size = Vec2i(width, height);
	int n3 = 3 * m_size.x*m_size.y;

	m_dx[frame] =  reinterpret_cast<Vec3f*>(dx);
	m_dy[frame] =  reinterpret_cast<Vec3f*>(dy);
	if (dt!=NULL)
		m_dt[frame] =  reinterpret_cast<Vec3f*>(dt);
	m_throughput[frame] =  reinterpret_cast<Vec3f*>(tp);


}

/* Marco: Make this Mitsuba friendlier. */
void Solver::importImagesMTS_GPT(float *dx, float *dy, float *tp, float *direct, int width, int height, int frame){

	m_size = Vec2i(width, height);
	int n3 = 3 * m_size.x*m_size.y;

	m_dx[frame] = reinterpret_cast<Vec3f*>(dx);
	m_dy[frame] = reinterpret_cast<Vec3f*>(dy);
	m_throughput[frame] = reinterpret_cast<Vec3f*>(tp);
    m_direct[frame] = reinterpret_cast<Vec3f*>(direct);

}

void Solver::evaluateMetricsMTS(float *err, float &errL1, float &errL2)
{
	assert(m_dx[0] && m_dy[0] && m_backend && m_x);
	assert(m_size.x > 0 && m_size.y > 0);
	int n = m_size.x * m_size.y;

	// Calculate L1 and L2 error.
	{
		m_backend->calc_Px(m_e, m_P, m_x, m_params.nframes);          // e = P*x
		m_backend->calc_axpy(m_e, -1.0f, m_e, m_b); // e = b - P*x
		const Vec3f* p_e = (const Vec3f*)m_backend->map(m_e);

		errL1 = 0.0f;
		errL2 = 0.0f;
		for (int i = 0; i < n * 3; i++)
		{
			errL1 += length(p_e[i]);
			errL2 += lenSqr(p_e[i]);
		}
		errL1 /= (float)(n * 3);
		errL2 /= (float)(n * 3);

		for (int i = 0; i < m_size.x*m_size.y; i++){
			err[3 * i] = p_e[i].x;
			err[3 * i + 1] = p_e[i].y;
			err[3 * i + 2] = p_e[i].z;
		}

		m_backend->unmap(m_e, (void*)p_e, false);
	}
}
void Solver::exportImagesMTS(float *rec, int f)
{
	assert(f<m_params.nframes);

	int n = m_size.x*m_size.y;
	// Indirect.
	{
		const Vec3f* p_x = (const Vec3f*)m_backend->map(m_x);
		//exportImage(p_x, m_params.indirectPFM.c_str(), m_params.indirectPNG.c_str(), false, m_z);
		
		for (int i = 0; i < n; i++){
		rec[3 * i]		= p_x[f*n + i].x;
		rec[3 * i + 1]  = p_x[f*n + i].y;
		rec[3 * i + 2]  = p_x[f*n + i].z;
		}

		m_backend->unmap(m_x, (void*)p_x, false);
		}

	// Final.
	{
        if (!m_direct[f])
			m_backend->copy(m_r, m_x);
		else
		{
            m_backend->write(m_r, m_direct[f]);
			m_backend->calc_axpy(m_r, 1.0f, m_r, m_x);
		}

		const Vec3f* p_r = (const Vec3f*)m_backend->map(m_r);
		//exportImage(p_r, m_params.finalPFM.c_str(), m_params.finalPNG.c_str(), false, m_z);

		for (int i = 0; i < m_size.x*m_size.y; i++){
			rec[3 * i] = p_r[f*n + i].x;
			rec[3 * i + 1] = p_r[f*n + i].y;
			rec[3 * i + 2] = p_r[f*n + i].z;
		}

		m_backend->unmap(m_r, (void*)p_r, false);
	}


}

void Solver::combineGradients(Vec3f *dx, Vec3f * dy, const Vec3f *const g0, const Vec3f *const g1, const Vec3f *const g2, const Vec3f *const g3, int kX, int kY)
{
	int n = m_size.x * m_size.y;
	for (int i = 0; i < n; ++i){
		
		dx[i] = i+kX<n		 ? 0.5f*(g2[i] - g1[i+kX]) : g2[i];
		dy[i] = i+kY<n ? 0.5f*(g3[i] - g0[i+kY]) : g3[i];
	}

}

void Solver::combineGradientsT(Vec3f *dx, Vec3f * dy, Vec3f* dt, const Vec3f *const g0, const Vec3f *const g1, const Vec3f *const g2, const Vec3f *const g3, const Vec3f *const g4, const Vec3f *const g5, int kX, int kY)
{
	int n = m_size.x * m_size.y;
	for (int i = 0; i < n; ++i){

		dx[i] = i + kX < n ? 0.5f*(g2[i] - g1[i + kX]) : g2[i];
		dy[i] = i + kY < n ? 0.5f*(g3[i] - g0[i + kY]) : g3[i];
		dt[i] =  g4 != NULL ? 0.5f*(g5[i] - g4[i]) :  g5[i];
		//if (dt[i].x != 0.f || dt[i].y != 0.f || dt[i].z != 0.f)
		//
		//printf("\nnon zero temporal gradient at position %d! (x=%f, y=%f, z=%f)", i, dt[i].x, dt[i].y, dt[i].z);
	}

}

void Solver::display(const char* title)
{
	// m_debugWindow.setTitle(title);
	// m_debugWindow.setSize(m_size.x, m_size.y);
	// m_backend->read(m_debugWindow.getPixelPtr(), m_tonemapped);
	// m_debugWindow.display();
}


void Solver::log(const std::string& message)
{
	m_params.logFunc(message);
}

//------------------------------------------------------------------------

void Solver::log(const char* fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	int len = _vscprintf(fmt, args);
	std::string str;
	str.resize(len);
	vsprintf_s((char*)str.c_str(), len + 1, fmt, args);
	va_end(args);

	log(str);
}

//------------------------------------------------------------------------


//------------------------------------------------------------------------

