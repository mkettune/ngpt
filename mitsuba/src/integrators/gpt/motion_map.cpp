#include "motion_map.h"
#include <limits.h>

MTS_NAMESPACE_BEGIN

MotionMap::MotionMap(int width, int height)
{
	m_width  = width;
	m_height = height;

	m_motionForward.resize(3 * width * height);
	m_motionBackward.resize(3 * width * height, std::numeric_limits<Float>::infinity());
}

MotionMap::MotionMap(Stream *stream, InstanceManager *manager)
{
	int npix = 3 * m_width * m_height;
	m_width = stream->readInt();
	m_height = stream->readInt();
	for (int i = 0; i < npix; i++)
		m_motionForward[i] = stream->readFloat();
}



void MotionMap::serialize(Stream *stream, InstanceManager *manager) const
{
	int npix = 3 * m_width*m_height;
	stream->writeInt(m_width);
	stream->writeInt(m_height);
	for (int i = 0; i < npix; i++)
		stream->writeFloat(m_motionForward[i]);
}

/// set complete frame
/// note mitsubas plugin is not capable of dealing with motion blur, hence filter the motion map if requested by the user
void MotionMap::setForwardMap(Float* data, float shutteropen, bool applyFilter){
	int npix = m_width*m_height;
	for (int i=0; i < npix; i++){
		m_motionForward[3 * i]	 = std::round(data[3 * i]);
		m_motionForward[3 * i + 1] = std::round(data[3 * i + 1]);
		m_motionForward[3 * i + 2] = data[3 * i + 2];
	}

	//compute backwards map too
	computeBackwardsMap();
}

void MotionMap::makeConservative(Float* reverseData){
	int npix = m_width * m_height;
	std::vector<float> reverseMotion(3 * npix, 0.0f);

	for (int i = 0; i < npix; i++) {
		reverseMotion[3 * i]	 = std::round(reverseData[3 * i]);
		reverseMotion[3 * i + 1] = std::round(reverseData[3 * i + 1]);
		reverseMotion[3 * i + 2] = reverseData[3 * i + 2];
	}

	// Kill motion vectors whose shifts' reverse-shift is not the original pixel.
	for (int i = 0; i < npix; ++i) {
		if(std::isinf(m_motionForward[3 * i + 0])) {
			continue;
		}

		int currentX = i % m_width;
		int currentY = i / m_width;

		int mappedX = currentX + (int)m_motionForward[3 * i + 0];
		int mappedY = currentY + (int)m_motionForward[3 * i + 1];

		Assert(mappedX >= 0 && mappedX < m_width && mappedY >= 0 && mappedY < m_height);

		int mappedI = m_width * mappedY + mappedX;

		int reverseMappedX = mappedX + (int)reverseMotion[3 * mappedI + 0];
		int reverseMappedY = mappedY + (int)reverseMotion[3 * mappedI + 1];

		// Accept a small tolerance.
		if(!(currentX - 1 <= reverseMappedX && reverseMappedX <= currentX + 1 &&
			 currentY - 1 <= reverseMappedY && reverseMappedY <= currentY + 1))
		{
			// Kill the motion vector.
			m_motionForward[3 * i + 0] = std::numeric_limits<float>::infinity(); 
			m_motionForward[3 * i + 1] = std::numeric_limits<float>::infinity();
			m_motionForward[3 * i + 2] = std::numeric_limits<float>::infinity();

			m_motionBackward[3 * mappedI + 0] = std::numeric_limits<float>::infinity();
			m_motionBackward[3 * mappedI + 1] = std::numeric_limits<float>::infinity();
			m_motionBackward[3 * mappedI + 2] = std::numeric_limits<float>::infinity();
			continue;
		}
	}
}


/// compute backwards correspondence map and clear forward map of non-injective entries
void MotionMap::computeBackwardsMap(){
	int npix = m_width*m_height;
	float xo, yo, d;
	int idx;

	//store inverted motion vectors in correspondence pixel of backward buffer, but only store the correspondence that is closest to the camera
	for (int i=0; i < npix; i++){
		int ix = i % m_width;
		int iy = i / m_width;

		xo = m_motionForward[3 * i];
		yo = m_motionForward[3 * i + 1];
		d  = m_motionForward[3 * i + 2];

		idx = (i + (int)(yo) * m_width + (int)xo);
		
		if (ix + (int)xo < 0 || ix + (int)xo >= m_width || iy + (int)yo < 0 || iy + (int)yo >= m_height)
			continue;

		if ((std::isinf(m_motionBackward[3 * idx]) || d < m_motionBackward[3 * idx + 2])){
			m_motionBackward[3 * idx] = -xo;
			m_motionBackward[3 * idx + 1] = -yo;
			m_motionBackward[3 * idx + 2] = d;
		}
	}
	
	//now do the whole thing again to erase non-injective entries in m_motionForward
	std::fill(m_motionForward.begin(), m_motionForward.end(), std::numeric_limits<Float>::infinity());
	for (int i=0; i < npix; i++){
		xo = m_motionBackward[3 * i];
		yo = m_motionBackward[3 * i + 1];
		d  = m_motionBackward[3 * i + 2];

		if (std::isfinite(xo)){
			idx = (i + (int)(yo) * m_width + (int)xo);
			m_motionForward[3 * idx] = -xo;
			m_motionForward[3 * idx + 1] = -yo;
			m_motionForward[3 * idx + 2] = d;
		}

	}
}

void MotionMap::backProjectBuffer(Float *input, Float *output){
	int npix = m_width*m_height;
	float xo, yo;
	int idx;

	for (int i=0; i < npix; i++){
		xo = m_motionForward[3 * i];
		yo = m_motionForward[3 * i + 1];

		idx = (i + yo*m_width + xo);
		
		// This is an unnecessary check, but it's nice not to segmentation fault in case of an invalid motion map.
		if (idx < 0 || idx >= npix)
			continue;

		output[3 * i] = input[3 * idx];
		output[3 * i + 1] = input[3 * idx + 1];
		output[3 * i + 2] = input[3 * idx + 2];

	}
}

/// set complete frame from half array (does the conversion work correctly?)
void MotionMap::setForwardMap(half* data){
	int npix = m_width*m_height;
	for (int i=0; i < npix; i++){
		m_motionForward[3 * i]	 = float(data[3 * i]);
		m_motionForward[3 * i + 1] = float(data[3 * i + 1]);
		m_motionForward[3 * i + 2] = float(data[3 * i + 2]);
	}
}

void MotionMap::setBackwardMap(half* data){
	int npix = m_width*m_height;
	for (int i=0; i < npix; i++){
		m_motionBackward[3 * i]	 = float(data[3 * i]);
		m_motionBackward[3 * i + 1] = float(data[3 * i + 1]);
		m_motionBackward[3 * i + 2] = float(data[3 * i + 2]);
	}
}

/// getFrame
float* MotionMap::getForwardMap(){
	return m_motionForward.data();
}

float* MotionMap::getBackwardMap(){
	return m_motionBackward.data();
}

/// getOffset
Point2i MotionMap::getOffset(int x, int y){
	int idx = y*m_width + x;

	//if no valid entry is in this pixel then
	//return "impossible" offset, such that it will definitely be out of the image plane.
	//(mitsuba will then handle all the rest by itself)
	if (std::isinf(m_motionForward[3 * idx]) || std::isinf(m_motionForward[3 * idx + 1]))
		return Point2i(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());

	return Point2i(int(m_motionForward[3*idx]), int(m_motionForward[3*idx+1]));
}

/// getOffset
Point2i MotionMap::getBackwardOffset(int x, int y){
	int idx = y*m_width + x;

	//if no valid entry is in this pixel then
	//return "impossible" offset, such that it will definitely be out of the image plane.
	//(mitsuba will then handle all the rest by itself)
	if (std::isinf(m_motionBackward[3 * idx]) || std::isinf(m_motionBackward[3 * idx + 1]))
		return Point2i(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());

	return Point2i(int(m_motionBackward[3 * idx]), int(m_motionBackward[3 * idx + 1]));
}


MTS_NAMESPACE_END
