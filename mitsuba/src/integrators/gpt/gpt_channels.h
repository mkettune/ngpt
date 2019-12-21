/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#if !defined(__GPT_CHANNELS_H)
#define __GPT_CHANNELS_H

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN


namespace gpt {


// Output buffer names.
static const size_t BUFFER_FINAL = 0;       ///< Buffer index for the final image. Also used for preview.
static const size_t BUFFER_THROUGHPUT = 1;  ///< Buffer index for the noisy color image.
static const size_t BUFFER_DX = 2;          ///< Buffer index for the X gradients.
static const size_t BUFFER_DY = 3;          ///< Buffer index for the Y gradients.

static const size_t BUFFER_VERY_DIRECT = 4; ///< Buffer index for very direct light.

static const size_t SAMPLING_MAP_VARIANCE_TP = 5;  ///< Buffer index for the noisy color image variance.
static const size_t SAMPLING_MAP_VARIANCE_DX = 6;  ///< Buffer index for the X gradients variance.
static const size_t SAMPLING_MAP_VARIANCE_DY = 7;  ///< Buffer index for the Y gradients variance.

static const size_t FEAT_ALBEDO = 8;
static const size_t FEAT_NORMAL = 9;
static const size_t FEAT_POSITION = 10;
static const size_t FEAT_DEPTH = 11;
static const size_t FEAT_VISIBILITY = 12;

static const size_t FEAT_VAR_ALBEDO = 13;
static const size_t FEAT_VAR_NORMAL = 14;
static const size_t FEAT_VAR_POSITION = 15;
static const size_t FEAT_VAR_DEPTH = 16;
static const size_t FEAT_VAR_VISIBILITY = 17;

static const size_t FEAT_VAR_PRIMAL = 18; // Separate buffers to not mess up with the sampling map code.
static const size_t FEAT_VAR_DX = 19;
static const size_t FEAT_VAR_DY = 20;

static const size_t FEAT_DIFFUSE = 21;
static const size_t FEAT_SPECULAR = 22;
static const size_t FEAT_VAR_DIFFUSE = 23;
static const size_t FEAT_VAR_SPECULAR = 24;

static const size_t BUFFER_COUNT = 25;


static const size_t FEATURE_COUNT = 7; // Note: Also update gpt.cpp:renderBlock.

inline std::vector<std::string> bufferNames() {
	return {
		"-motion",
		"-primal",
		"-dx",
		"-dy",
		"-direct",
		"-sampling-map-var-primal",
		"-sampling-map-var-dx",
		"-sampling-map-var-dy",
		"-albedo",
		"-normal",
		"-position",
		"-depth",
		"-visibility",
		"-var-albedo",
		"-var-normal",
		"-var-position",
		"-var-depth",
		"-var-visibility",
		"-var-primal",
		"-var-dx",
		"-var-dy",
		"-diffuse",
		"-specular",
		"-var-diffuse",
		"-var-specular",
		"-spp",
		"-motion-inv",
		"-final",
	};	
}


}


MTS_NAMESPACE_END


#endif /* __GPT_CHANNELS_H */
