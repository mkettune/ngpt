#if !defined(__SMAP_SM)
#define __SMAP_SM

#include "mitsuba/core/serialization.h"
#include "mitsuba/core/bitmap.h"
#include "mitsuba/mitsuba.h"
#include "mitsuba/core/random.h"

MTS_NAMESPACE_BEGIN

class SamplingMap : public SerializableObject {
public:
	SamplingMap(int initial_spp, int max_spp, int width, int height, int seed=0, bool only_gradients=true);

	/// Unserialize a serializable object
	SamplingMap(Stream *stream, InstanceManager *manager);

	/// Serialize this object to a stream
	void serialize(Stream *stream, InstanceManager *manager) const;

	///  computes a sampling-map that uses blurred variance maps normalized by the throughputs luminance as guide
	bool computeMapWithGradients(Float *prim_var, Float *dx_var, Float *dy_var, Float *lum, Float *lum_dx, Float *lum_dy, int requested_spp);

	/// sets map based on a bitmap. Should only be used for offset paths
	void setSamplingMap(half *data);

	/// process sampling map with median filter followed by max filter. I.e. m_tmp = max_flt(median_flt(m_tmp, rad_median), rad_max)
	void processEnergy(int rad_median, int rad_max);

	void updateAccum();

	/// Simple getter function for map values
	int get(int x, int y);

	/// Simple getter function for map values
	int getAccum(int x, int y);

	//// returns a visualization of the currently stored sampling-map as a bitmap. Method is slow, hence it should be used for debugging only
	void getImageAccum(Bitmap *bitmap);

	void getImage(Bitmap *bitmap);

private:
	/// 
	Float splatInSPP(const Float &energy, Float *weights, int len);

	/// Generates a integer map from m_spp_tmp and redistributes n free samples uniformly. And returns the new mean.
	Float redistributeUniformly(const Float &free_spp);

	std::vector<int> m_map;				//current samples that should be distributed
	std::vector<Float> m_spp;			//temporary data. stores float-value spp. never access this directly from outside!
	std::vector<int> m_accum_map;		//accumulated spp so far. used to compute variance of mean

	std::vector<Float> m_tmp, m_tmp2;	//temporary data. helpful for map computation

	ref<Random> m_rgn;					//random generator
	Vector2i m_size;					//size of image
	int m_maxspp;						//max spp per iteration, used to limit execution-time of worker
	int m_requested_spp;				//requested average spp per iteration. 
	bool m_only_gradients;              //whether to use only gradients for the sample distribution
};

MTS_NAMESPACE_END
#endif
