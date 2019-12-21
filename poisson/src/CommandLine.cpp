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

#include "CommandLine.hpp"
#include "Solver.hpp"
#include <stdio.h>

using namespace poisson;

//------------------------------------------------------------------------

void printUsage(void)
{
    printf("\n");
    printf("Usage: poisson.exe [OPTIONS]\n");
    printf("Reconstruct an image (or a sequence of images) from its spatial (and temporal) gradients by solving the screened Poisson equation.\n");
	printf("Base code: Tero Karras (tkarras@nvidia.com)\nTemporal extension: Marco Manzi (manzi@iam.unibe.ch)\n");
    printf("\n");
	printf("Temporal GPT specific settings:\n");
	printf("  -mv                    (IMPORTANT) Must be enabled if Mitsuba used motion-vectors for TGPT!\n");
	printf("  -time                  Use temporal gradients.\n");
	printf("  -ddxt                  Use spatio-temporal gradients.\n");
	printf("  -weight_dt    <1>      How much weight to put on the temporal gradients.\n");
	printf("  -weight_dsdt  <1>      How much weight to put on the spatio-temporal gradients.\n");
	printf("\n");
    printf("Input images in PFM format:\n");
    printf("  -dx           <PFM>    Input for dx. Typically '<BASE>-dx.pfm'. (All other inputs must be in <BASE>!)\n");
	printf("  -nframes      <1>      Number of frames to reconstruct at once.\n");
	printf("  -iframe       <0>      Index of first frame.\n");
	printf("  -alpha        <0.2>    How much weight to put on the throughput image compared to the gradients.\n");
	printf("  -weight_ds    <1>      How much weight to put on the spatial gradients.\n");
	printf("\n");
    printf("PNG conversion:\n");
    printf("  -brightness 1.0    Scale image intensity before converting to sRGB color space.\n");
    printf("  -nopngout          Do not convert output images to PNG.\n");
    printf("\n");
    printf("Other options:\n");
    printf("  -backend  CUDA     Enable GPU acceleration using CUDA. Requires a GPU with compute capability 3.0 or higher.\n");
    printf("  -backend  OpenMP   Enable multicore acceleration using OpenMP.\n");
    printf("  -backend  Naive    Use naive single-threaded CPU implementation.\n");
    printf("  -backend  Auto     Use 'CUDA' if available, or fall back to 'OpenMP' if not. This is the default.\n");
    printf("  -device   0        Choose the CUDA device to use. Only applicable to 'CUDA' and 'Auto'.\n");
    printf("  -verbose, -v       Enable verbose printouts.\n");
    printf("  -display, -d       Display progressive image refinement during the solver.\n");
    printf("  -help,    -h       Display this help text.\n");
    printf("\n");
    printf("Solver presets (default is L1D):\n");
    printf("  -config  L1D      L1 default config: ~1s for 1280x720 on GTX980, L1 error lower than MATLAB reference.\n");
    printf("  -config  L1Q      L1 high-quality config: ~50s for 1280x720 on GTX980, L1 error as low as possible.\n");
    printf("  -config  L1L      L1 legacy config: ~89s for 1280x720 on GTX980, L1 error equal to MATLAB reference.\n");
    printf("  -config  L2D      L2 default config: ~0.1s for 1280x720 on GTX980, L2 error equal to MATLAB reference.\n");
    printf("  -config  L2Q      L2 high-quality config: ~0.5s for 1280x720 on GTX980, L2 error as low as possible.\n");
    printf("\n");
    printf("Solver configuration:\n");
    printf("  -irlsIterMax 20   Number of iteratively reweighted least squares (IRLS) iterations.\n");
    printf("  -irlsRegInit 0.05 Initial value of the IRLS regularization parameter.\n");
    printf("  -irlsRegIter 0.5  Multiplier for the IRLS regularization parameter on subsequent iterations.\n");
    printf("  -cgIterMax   50   Maximum number of conjugate gradient (CG) iterations per IRLS iteration.\n");
    printf("  -cgIterCheck 100  Check status every N iterations (incl. early exit, CPU-GPU sync, printouts, image display).\n");
    printf("  -cgPrecond   0    0 = regular conjugate gradient (optimized), 1 = preconditioned conjugate gradient (experimental).\n");
    printf("  -cgTolerance 0    Stop CG iteration when the weight L2 error  (errL2\n");
    printf("\n");
    printf("Example:\n");
    printf("  poisson.exe -dx scenes/bathroom-dx.pfm -alpha 0.2 -brightness 2\n");
}

//------------------------------------------------------------------------

bool fileExists(const std::string& path)
{
    FILE* f = NULL;
    fopen_s(&f, path.c_str(), "rb");
    if (f)
        fclose(f);
    return (f != NULL);
}

//------------------------------------------------------------------------

void fixInputImagePaths(std::string& pfmPath, std::string& pngPath, const char* suffix, const std::string& basePath, bool pngIn, bool noPngIn)
{
    // No PFM path specified => generate one from base path.

    if (!pfmPath.length() && basePath.length())
    {
        std::string path = basePath + suffix + ".pfm";
        if (fileExists(path))
            pfmPath = path;
    }

    // No PNG path specified => generate one from PFM path.

    if (pfmPath.length() && !pngPath.length())
        pngPath = pfmPath.substr(0, pfmPath.rfind('.')) + ".png";

    // Do we want to create the PNG?

    if (!pfmPath.length() || noPngIn || (!pngIn && pngPath.length() && fileExists(pngPath)))
        pngPath = "";
}

//------------------------------------------------------------------------

void fixOutputImagePaths(std::string& pfmPath, std::string& pngPath, const char* suffix, const std::string& basePath, bool noOut, bool pngOut)
{
    // Disabled => do not output PFM.
    // PFM path not specified => generate one from base path.

    if (noOut)
        pfmPath = "";
    else if (!pfmPath.length() && basePath.length())
        pfmPath = basePath + suffix + ".pfm";

    // Disabled => do not output PNG.
    // PNG path not specified => generate one from PFM path.

    if (!pfmPath.length() || !pngOut)
        pngPath = "";
    else if (!pngPath.length())
        pngPath = pfmPath.substr(0, pfmPath.rfind('.')) + ".png";
}

//------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    // Parse command line.
    Solver::Params p;
    bool        noIndirect  = false;
    bool        noFinal     = false;
    bool        pngIn       = false;
    bool        noPngIn     = false;
    bool        pngOut      = true;
    bool        help        = false;
	bool		frameSuffix	= false;
    std::string error       = "";

	//default values
	p.alpha		= 0.2f;
	p.nframes	= 1;
	p.iframe	= 0;

    for (int i = 1; i < argc; i++)
    {
             if (strcmp(argv[i], "-dx")         == 0 && i + 1 < argc)   p.dxPFM[0]			= argv[++i];
        else if (strcmp(argv[i], "-direct")     == 0 && i + 1 < argc)   p.directPFM[0]		= argv[++i];
        else if (strcmp(argv[i], "-reference")  == 0 && i + 1 < argc)   p.referencePFM		= argv[++i];

		else if (strcmp(argv[i], "-brightness") == 0 && i + 1 < argc)   p.brightness		= std::stof(argv[++i]);
		else if (strcmp(argv[i], "-nopngout") == 0)						pngOut				= false;
        else if (strcmp(argv[i], "-backend")    == 0 && i + 1 < argc)   p.backend			= argv[++i];
        else if (strcmp(argv[i], "-device")     == 0 && i + 1 < argc)   p.cudaDevice		= std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-verbose")    == 0)                   p.verbose			= true;
        else if (strcmp(argv[i], "-v")          == 0)                   p.verbose			= true;
        else if (strcmp(argv[i], "-display")    == 0)                   p.display			= true;
        else if (strcmp(argv[i], "-d")          == 0)                   p.display			= true;
        else if (strcmp(argv[i], "-help")       == 0)                   help				= true;
        else if (strcmp(argv[i], "-h")          == 0)                   help				= true;
        else if (strcmp(argv[i], "-config")     == 0 && i + 1 < argc)
        {
            const char* preset = argv[++i];
            if (!p.setConfigPreset(preset))
                if (!error.length())
                    error = poisson::sprintf("Invalid config preset '%s'!", preset);
        }
        else if (strcmp(argv[i], "-irlsIterMax")    == 0 && i + 1 < argc)   p.irlsIterMax   = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-irlsRegInit")    == 0 && i + 1 < argc)   p.irlsRegInit   = std::stof(argv[++i]);
        else if (strcmp(argv[i], "-irlsRegIter")    == 0 && i + 1 < argc)   p.irlsRegIter   = std::stof(argv[++i]);
        else if (strcmp(argv[i], "-cgIterMax")      == 0 && i + 1 < argc)   p.cgIterMax     = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-cgIterCheck")    == 0 && i + 1 < argc)   p.cgIterCheck   = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-cgPrecond")      == 0 && i + 1 < argc)   p.cgPrecond     = (std::stoi(argv[++i]) != 0);
        else if (strcmp(argv[i], "-cgTolerance")    == 0 && i + 1 < argc)   p.cgTolerance   = std::stof(argv[++i]);

		//parames for TGPT
		else if (strcmp(argv[i], "-time")			== 0)				p.useTime			= true;
		else if (strcmp(argv[i], "-ddxt")			== 0)				p.useDDxt			= true; 
		else if (strcmp(argv[i], "-mv")				== 0)				p.useMotion			= true && p.useTime; 
		else if (strcmp(argv[i], "-nframes") == 0 && i + 1 < argc){		p.nframes			= std::stoi(argv[++i]);  frameSuffix = true; }
		else if (strcmp(argv[i], "-iframe") == 0 && i + 1 < argc){		p.iframe			= std::stoi(argv[++i]);  frameSuffix = true; }

		//weights of the equation system.
		else if (strcmp(argv[i], "-alpha") == 0 && i + 1 < argc)		p.alpha				= std::stof(argv[++i]);
		else if (strcmp(argv[i], "-weight_ds") == 0 && i + 1 < argc)	p.w_ds				= std::stof(argv[++i]);
		else if (strcmp(argv[i], "-weight_dt") == 0 && i + 1 < argc)    p.w_dt				= std::stof(argv[++i]);
		else if (strcmp(argv[i], "-weight_dsdt") == 0 && i + 1 < argc)  p.w_dsdt			= std::stof(argv[++i]);
        else
        {
            if (!error.length())
                error = poisson::sprintf("Invalid command line option '%s'!", argv[i]);
        }
    }


	printf("Number of frames: %d \nStarting frame: %d\n\n", p.nframes, p.iframe);


    std::string basePath = p.dxPFM[0];
    basePath = basePath.substr(0, max((int)basePath.length() - (int)strlen("-dx.pfm"), 0));
    if (p.dxPFM[0] != basePath + "-dx.pfm")
        basePath = "";


	// if no iframe or nframes is set, assume files do not have a frame suffix
	char name[256];

	for (int f=0, fw = p.iframe; f < p.nframes; f++, fw++){

		p.g0PFM.resize(p.nframes);
		if (frameSuffix)	sprintf_s(name, "-dx_%d", fw);	else	sprintf_s(name, "-dx");
		fixInputImagePaths(p.g0PFM[f], p.g0PNG, name, basePath, pngIn, noPngIn);
		error += p.g0PFM[f].length() ? "" : std::string(name) + "  not found! \n";

		p.g1PFM.resize(p.nframes);
		if (frameSuffix)	sprintf_s(name, "-dy_%d", fw);	else	sprintf_s(name, "-dy");
		fixInputImagePaths(p.g1PFM[f], p.g1PNG, name, basePath, pngIn, noPngIn);
		error += p.g1PFM[f].length() ? "" : std::string(name) + "  not found! \n";

		if (p.useTime)
		{
			p.g2PFM.resize(p.nframes);
			if (frameSuffix)	sprintf_s(name, "-dt_%d", fw);	else	sprintf_s(name, "-dt");
			fixInputImagePaths(p.g2PFM[f], p.g2PNG, name, basePath, pngIn, noPngIn);
			error += p.g2PFM[f].length() ? "" : std::string(name) + "  not found! \n";
		}

		if (p.useDDxt)
		{
			p.ddxtPFM.resize(p.nframes);
			if (frameSuffix)	sprintf_s(name, "-ddxt_%d", fw);	else	sprintf_s(name, "-ddxt");
			fixInputImagePaths(p.ddxtPFM[f], p.ddxtPNG, name, basePath, pngIn, noPngIn);
			error += p.ddxtPFM[f].length() ? "" : std::string(name) + "  not found! \n";

			p.ddytPFM.resize(p.nframes);
			if (frameSuffix)	sprintf_s(name, "-ddyt_%d", fw);	else	sprintf_s(name, "-ddyt");
			fixInputImagePaths(p.ddytPFM[f], p.ddytPNG, name, basePath, pngIn, noPngIn);
			error += p.ddytPFM[f].length() ? "" : std::string(name) + "  not found! \n";
		}

		p.throughputPFM.resize(p.nframes);
		if (frameSuffix)	sprintf_s(name, "-primal_%d", fw);	else	sprintf_s(name, "-primal");
		fixInputImagePaths(p.throughputPFM[f], p.throughputPNG, name, basePath, pngIn, noPngIn);
		error += p.throughputPFM[f].length() ? "" : std::string(name) + "  not found! \n";

		if (p.useMotion)
		{
			p.motionPFM.resize(p.nframes);
			if (frameSuffix)	sprintf_s(name, "-motion_%d", fw);	else	sprintf_s(name, "-motion");
			fixInputImagePaths(p.motionPFM[f], p.motionPNG, name, basePath, pngIn, noPngIn);
			error += p.motionPFM[f].length() ? "" : std::string(name) + "  not found! \n";

			p.motionBW_PFM.resize(p.nframes);
			if (frameSuffix)	sprintf_s(name, "-motion-inv_%d", fw);	else	sprintf_s(name, "-motion-inv");
			fixInputImagePaths(p.motionBW_PFM[f], p.motionBW_PNG, name, basePath, pngIn, noPngIn);
			error += p.motionBW_PFM[f].length() ? "" : std::string(name) + "  not found! \n";
		}

		p.finalPFM.resize(p.nframes);
		p.finalPNG.resize(p.nframes);
		if (frameSuffix)	sprintf_s(name, "-final_%d", fw);	else	sprintf_s(name, "-final");
		fixOutputImagePaths(p.finalPFM[f], p.finalPNG[f], name, basePath, noFinal, pngOut);

	}

    if (error.length() || help)
    {
        printUsage();
        if (error.length() && !help)
        {
            printf("\nError: \n%s\n", error.c_str());
            return 1;
        }
        return 0;
    }

    Solver solver(p);

    solver.importImages();
    solver.setupBackend();
    solver.solveIndirect();
    solver.exportImages();

    return 0;
}

//------------------------------------------------------------------------
