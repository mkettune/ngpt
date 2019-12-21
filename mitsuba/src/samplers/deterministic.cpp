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

#include <mitsuba/render/sampler.h>

#define MAX_SEED 10000
#define MAX_FRAMES 1000 // TODO: Remove. We always render one frame at a time anyway.
#define DEBUG_RNG

MTS_NAMESPACE_BEGIN

/*!\plugin{deterministic}{Deterministic sampler}
 * \order{1}
 * \parameters{
 *     \parameter{sampleCount}{\Integer}{
 *       Number of samples per pixel \default{4}
 *     }
 *	   \parameter{seed}{\Integer}{
 *       Seed of sampler.  \default{123}
 *     }
 * }
 *
 * \renderings{
 *     \unframedrendering{A projection of the first 1024 points
 *     onto the first two dimensions. Note the sample clumping.}{sampler_independent}
 * }
 *
 * The independent sampler produces a stream of independent and uniformly
 * distributed pseudorandom numbers. Internally, it relies on a fast SIMD version
 * of the Mersenne Twister random number generator \cite{Saito2008SIMD}.
 *
 * This is the most basic sample generator; because no precautions are taken to avoid
 * sample clumping, images produced using this plugin will usually take longer to converge.
 */
class DeterministicSampler : public Sampler {
public:
	DeterministicSampler() : Sampler(Properties()) { }

	DeterministicSampler(const Properties &props) : Sampler(props) {
		/* Number of samples per pixel when used with a sampling-based integrator */
		m_sampleCount = props.getSize("sampleCount", 4);
		m_random = new Random();
	}

	DeterministicSampler(Stream *stream, InstanceManager *manager)
	 : Sampler(stream, manager) {
		m_random = static_cast<Random *>(manager->getInstance(stream));
		m_res = Vector2i(stream);
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		Sampler::serialize(stream, manager);
		manager->serialize(stream, m_random.get());
		m_res.serialize(stream);

	}

	ref<Sampler> clone() {
		ref<DeterministicSampler> sampler = new DeterministicSampler();
		sampler->m_sampleCount = m_sampleCount;
		sampler->m_random = new Random(m_random);
		sampler->m_res = m_res;
		for (size_t i=0; i<m_req1D.size(); ++i)
			sampler->request1DArray(m_req1D[i]);
		for (size_t i=0; i<m_req2D.size(); ++i)
			sampler->request2DArray(m_req2D[i]);
		return sampler.get();
	}

	void resetRNG(const Point2i & pos, int seed, int frame, int sample) {
		// Note: We don't necessarily need MAX_FRAMES here as we always render one frame per run anyway, with a changing seed.
		uint64_t rndseed = ((uint64_t)seed + (uint64_t)frame) +
			(uint64_t)MAX_SEED*(uint64_t)MAX_FRAMES*(uint64_t)pos.y +
			(uint64_t)MAX_SEED*(uint64_t)MAX_FRAMES*(uint64_t)m_res.y*(uint64_t)pos.x +
			(uint64_t)MAX_SEED*(uint64_t)MAX_FRAMES*(uint64_t)m_res.y*(uint64_t)m_res.x*(uint64_t)sample;

		m_random = new Random(rndseed);
	}

	void generate(const Point2i &pos) {
		for (size_t i=0; i<m_req1D.size(); i++)
			for (size_t j=0; j<m_sampleCount * m_req1D[i]; ++j)
				m_sampleArrays1D[i][j] = m_random->nextFloat();
		for (size_t i=0; i<m_req2D.size(); i++)
			for (size_t j=0; j<m_sampleCount * m_req2D[i]; ++j)
				m_sampleArrays2D[i][j] = Point2(
					m_random->nextFloat(),
					m_random->nextFloat());
		m_sampleIndex = 0;
		m_dimension1DArray = m_dimension2DArray = 0;
	}

	Float next1D() {
		return m_random->nextFloat();
	}

	Point2 next2D() {
		Float value1 = m_random->nextFloat();
		Float value2 = m_random->nextFloat();
		return Point2(value1, value2);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "IndependentSampler[" << endl
			<< "  sampleCount = " << m_sampleCount << endl
			<< "]";
		return oss.str();
	}

	void setFilmResolution(const Vector2i &res, bool blocked){
		m_res=res;
	}

	MTS_DECLARE_CLASS()
private:
	ref<Random> m_random;
	Vector2i m_res;
};

MTS_IMPLEMENT_CLASS_S(DeterministicSampler, false, Sampler)
MTS_EXPORT_PLUGIN(DeterministicSampler, "Deterministic sampler");
MTS_NAMESPACE_END
