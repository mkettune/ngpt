#if !defined(__SMAP_AM)
#define __SMAP_AM

#include "mitsuba/core/serialization.h"
#include "mitsuba/core/bitmap.h"
#include "mitsuba/mitsuba.h"

MTS_NAMESPACE_BEGIN
/*
* This class is wrapper for motion vector maps. It assumes to get (X,Y) values for correspondences and
* depth values for the primary hit as Z value. (motion.cpp has been modified for this)
* It is assumed to be used in the following way:
*  - The base pass generates this map by calculating the motion from the current frame to the next frame.
*		i.e given pixel at frame i it computes the correspondence at frame i+1
*  - In the time-offset passes the inverse maps are required, 
*		i.e. given pixel at frame i+1 where is its correspondence at frame i?
*		> these inverse maps are computed and stored on the fly.
*
*
*
*
*
*
*
**/
class MotionMap : public SerializableObject {
public:
	MotionMap(int width, int height);

	/// Unserialize a serializable object
	MotionMap(Stream *stream, InstanceManager *manager);

	/// Serialize this object to a stream
	void serialize(Stream *stream, InstanceManager *manager) const;

	/// set complete frame. Rounds first 2 components to next integer (since shifts are discretized by pixels)
	void setForwardMap(Float* data, float shutteropen=0, bool applyfilter=false);
	void setForwardMap(half* data);

	/// Kills motion vectors for which inversion of the forward motion map doesn't lead (close) to the original pixel.
	void makeConservative(Float* reverseData);

	/// back project buffers onto pixel-correspondence of last frame
	void backProjectBuffer(Float *input, Float *output);

	//void setBackwardMap(Float* data);
	void setBackwardMap(half* data);

	/// getFrame
	float* getForwardMap();
	float* getBackwardMap();
	
	/// getOffset
	Point2i getOffset(int x, int y);
	Point2i getBackwardOffset(int x, int y);

private:
	void computeBackwardsMap();

	std::vector<float> m_motionForward; 
	std::vector<float> m_motionBackward;
	int m_width;
	int m_height;
};

MTS_NAMESPACE_END
#endif
