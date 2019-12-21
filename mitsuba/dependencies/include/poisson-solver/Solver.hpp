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

#pragma once
#include "Backend.hpp"
//#include "DebugWindow.hpp"

#include <functional>
#include <vector>

namespace poisson
{
//------------------------------------------------------------------------
// Poisson solver main class.
//------------------------------------------------------------------------

class Solver
{
public:
    struct Params
    {
        // Input PFM files.

        std::vector<std::string> g0PFM;
        std::vector<std::string> g1PFM;
		std::vector<std::string> g2PFM;
		std::vector<std::string> g3PFM;
		std::vector<std::string> g4PFM;
		std::vector<std::string> g5PFM;

		std::string         g0PFM2;
		std::string         g1PFM2;
		std::string         g2PFM2;
		std::string         g3PFM2;

		std::vector<std::string> dxPFM;
		std::vector<std::string> dyPFM;
		std::vector<std::string> dtPFM;

		//second derivatives
		std::vector<std::string> ddxtPFM;
		std::vector<std::string> ddytPFM;
		std::vector<std::string> ddxxPFM;
		std::vector<std::string> ddyyPFM;
		std::vector<std::string> ddxyPFM;

        std::vector<std::string> throughputPFM;
		std::vector<std::string> motionPFM;
		std::vector<std::string> motionBW_PFM;

        std::vector<std::string> directPFM;
        std::string         referencePFM;
        float               alpha;
		float				w_ds;
		float               w_dt;
		float               w_dsdt;

		int					nframes;
		int					iframe;

        // Output PFM files.

        std::string         indirectPFM;
        std::vector<std::string>   finalPFM;

        // Output PNG files.

		std::string         g0PNG;
		std::string         g1PNG;
		std::string         g2PNG;
		std::string         g3PNG;
		std::string         g4PNG;
		std::string         g5PNG;

		std::string         g0PNG2;
		std::string         g1PNG2;
		std::string         g2PNG2;
		std::string         g3PNG2;

        std::string         dxPNG;
        std::string         dyPNG;

		//second derivatives
		std::string         ddxtPNG;
		std::string         ddytPNG;
		std::string         ddxxPNG;
		std::string         ddyyPNG;
		std::string         ddxyPNG;

        std::string         throughputPNG;
		std::string         motionPNG;
		std::string         motionBW_PNG;

        std::string         directPNG;
        std::string         referencePNG;
        std::string         indirectPNG;
        std::vector<std::string>         finalPNG;
        float               brightness;

        // Other parameters.

        std::string         backend;        // "Auto", "CUDA", "OpenMP", "Naive".
        int                 cudaDevice;
        bool                verbose;
        bool                display;

		bool                useTime;

		bool                useDDxt;
		bool				useDDxx;
		bool				useDDxy;

		bool                useMotion;
		bool				energy_weights;
		bool				L2;

        // Solver configuration.

        int                 irlsIterMax;
        float               irlsRegInit;
        float               irlsRegIter;
        int                 cgIterMax;
        int                 cgIterCheck;
        bool                cgPrecond;
        float               cgTolerance;

		// Debugging and information output.
		typedef std::function<void(const std::string&)> LogFunction;
		LogFunction         logFunc;

        // Methods.
                            Params          (void);
        void                setDefaults     (void);
        bool                setConfigPreset (const char* preset); // "L1D", "L1Q", "L1L", "L2D", "L2Q".
        void                sanitize        (void);

		void                setLogFunction(LogFunction function);
    };

public:
	void					importImagesMTS(float *dx, float *dy, float* dt, float *tp, int width, int height, int frame=0);
	void					importImagesMTS_GPT(float *dx, float *dy, float *tp, float *direct, int width, int height, int frame=0); //overloaded for gpt
	void					evaluateMetricsMTS(float *err, float &errL1, float &errL2); //not supported
	void                    exportImagesMTS(float *rec, int f=0); 

	Solver(const Params& params);
                            ~Solver         (void);

    void                    importImages    (void);
    void                    setupBackend    (void);
    void                    solveIndirect   (void);
    void                    evaluateMetrics (void);
    void                    exportImages    (void);

    void					setImages		(const float * const h_ib, const float * const h_dx, const float * const h_dy, const int width, const int height);
    void					getImage		(float *h_dst);

private:
    void                    importImage     (Vec3f** pixelPtr, const char* pfmPath);
    void                    exportImage     (const Vec3f* pixelPtr, const char* pfmPath, const char* pngPath, bool gradient, Backend::Vector* tmp);
	void                    exportImage2(const Vec3f* pixelPtr, std::vector<std::string> pfmPath, std::vector<std::string> pngPath, Backend::Vector* tmp);
	void					exportImageDebug(const Vec3f* pixelPtr, const char* pngPath, Backend::Vector* tmp);
    void                    display         (const char* title);

private:
	void                    log(const std::string& message);
	void                    log(const char* fmt, ...);

	void					combineGradients(Vec3f *dx, Vec3f * dy, const Vec3f *const g0, const Vec3f *const g1, const Vec3f *const g2, const Vec3f *const g3, int kX, int kY);
	void					combineGradientsT(Vec3f *dx, Vec3f * dy, Vec3f * dt, const Vec3f *const g0, const Vec3f *const g1, const Vec3f *const g2, const Vec3f *const g3, const Vec3f *const g4, const Vec3f *const g5, int kX, int kY);
private:
                            Solver          (const Solver&); // forbidden
    Solver&                 operator=       (const Solver&); // forbidden

private:
    Params                  m_params;
//    DebugWindow             m_debugWindow;

    Vec2i                   m_size;

	std::vector<Vec3f*>		m_dx;
	std::vector<Vec3f*>		m_dy;
	std::vector<Vec3f*>		m_dt;
	std::vector<Vec3f*>		m_ddxt;
	std::vector<Vec3f*>		m_ddyt;
	std::vector<Vec3f*>		m_ddxx;
	std::vector<Vec3f*>		m_ddyy;
	std::vector<Vec3f*>		m_ddxy;
	std::vector<Vec3f*>		m_g0;
	std::vector<Vec3f*>		m_g1;
	std::vector<Vec3f*>		m_g2;
	std::vector<Vec3f*>		m_g3;
	std::vector<Vec3f*>		m_g4;
	std::vector<Vec3f*>		m_g5;


	Vec3f*                  m_dx2;
	Vec3f*                  m_dy2;

	Vec3f*                  m_g0_2;
	Vec3f*                  m_g1_2;
	Vec3f*                  m_g2_2;
	Vec3f*                  m_g3_2;

 //   Vec3f*                  m_throughput;
	std::vector<Vec3f*>     m_throughput;
	std::vector<Vec3f*>     m_motion;
	std::vector<Vec3f*>     m_motion_bw;

    std::vector<Vec3f*>     m_direct;
    Vec3f*                  m_reference;

    Backend*                m_backend;
    Backend::PoissonMatrix  m_P;
    Backend::Vector*        m_b;
	Backend::Vector*        m_m; //motion vectors
	Backend::Vector*        m_mbw; //motion vectors backwards
    Backend::Vector*        m_e;
    Backend::Vector*        m_w2;
    Backend::Vector*        m_x;
    Backend::Vector*        m_r;
    Backend::Vector*        m_z;
	Backend::Vector*        m_tmpImg;
    Backend::Vector*        m_p;
    Backend::Vector*        m_Ap;
    Backend::Vector*        m_rr;
    Backend::Vector*        m_rz;
    Backend::Vector*        m_rz2;
    Backend::Vector*        m_pAp;
    Backend::Vector*        m_tonemapped;
	Backend::Vector*        m_tonemapped_small;
    Backend::Timer*         m_timerTotal;
    Backend::Timer*         m_timerIter;
};

//------------------------------------------------------------------------
}
