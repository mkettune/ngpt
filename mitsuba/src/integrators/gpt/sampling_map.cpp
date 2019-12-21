#include "sampling_map.h"

MTS_NAMESPACE_BEGIN

SamplingMap::SamplingMap(int initial_spp, int max_spp, int width, int height, int seed, bool only_gradients)
{
	m_size = Vector2i(width, height);;
	m_maxspp = max_spp;
	m_requested_spp = initial_spp;
	m_only_gradients = only_gradients;
	m_rgn = new Random(seed + 4298456); //magic number

	int npix = width * height;
	m_tmp.resize(npix, Float(0.0));
	m_tmp2.resize(npix, Float(0.0));

	m_spp.resize(npix, Float(0.0));;
	m_map.resize(npix, initial_spp);
	m_accum_map.resize(npix, 0/*initial_spp*/);

	//DEBUG ONLY: fill map with fake chess-board data 
	/*int d = 40;
	float f = 0.25f;
	for (int i = 0; i < npix; i++){
	int y = i / m_size.x;
	int x = i - y*m_size.x;
	if ((x % d < d/2 && y % d >=d/2) || (x % d >=d/2 && y % d <d/2))
	m_map[i] *= f;
	else
	m_map[i] *= 2.f-f;
	}*/
}


void SamplingMap::setSamplingMap(half *data){
	int npix = m_size.x * m_size.y;
	for (int i=0; i < npix; i++){
		m_map[i] = data[3*i]; //data is RGB but all channels are the same!
		//m_accum_map[i] = m_map[i];
	}
}

void SamplingMap::updateAccum(){
	int npix = m_size.x * m_size.y;
	for (int i=0; i < npix; i++)
		m_accum_map[i] += m_map[i]; //add samples to accumulated map
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
* Slightly more complicated. Scales value by second data buffer (usually luminance of pixel)
*/
bool SamplingMap::computeMapWithGradients(Float *prim_var, Float *dx_var, Float *dy_var, Float *lum, Float *lum_dx, Float *lum_dy, int requested_spp)
{

	Log(EWarn, "starting to compute sampling map for next iteration...");

	m_requested_spp = requested_spp;

	int npix = m_size.x * m_size.y;
	Float energy = 0;


	//parameters
	Float alpha = 0.2; //weight for importance of base vs. gradients
	int maxit = 3;	//iteration number for redistribution of residual samples. 3 are enough for most scenes
	Float forced_free = 1.0;

	Float epsilon = 10e-4;
	//compute total energy of data
	for (int i = 0; i < npix; i++){
		m_map[i] = 0; //clear content in map
		m_spp[i] = 0; //clear content in map

		//prefilter Luminance
		Float w[] ={ 0.2, 0.2, 0.2, 0.2, 0.2 };
		int len = 2;

		Float l = 0.0, l_x = 0.0, l_y = 0.0;

		int y = i / m_size.x;
		int x = i - y*m_size.x;
		if (y > len - 1 && y < m_size.y - len && x > len - 1 && x < m_size.x - len)
		{
			for (int xo = -len; xo <= len; xo++)
			{
				for (int yo = -len; yo <= len; yo++){
					int idx_o = i + yo*m_size.x + xo;
					l += w[xo + len] * w[yo + len] * lum[3 * idx_o];
					l += w[xo + len] * w[yo + len] * lum[3 * idx_o + 1];
					l += w[xo + len] * w[yo + len] * lum[3 * idx_o + 2];

					l_x += w[xo + len] * w[yo + len] * lum_dx[3 * idx_o];
					l_x += w[xo + len] * w[yo + len] * lum_dx[3 * idx_o + 1];
					l_x += w[xo + len] * w[yo + len] * lum_dx[3 * idx_o + 2];

					l_y += w[xo + len] * w[yo + len] * lum_dy[3 * idx_o];
					l_y += w[xo + len] * w[yo + len] * lum_dy[3 * idx_o + 1];
					l_y += w[xo + len] * w[yo + len] * lum_dy[3 * idx_o + 2];
				}
			}
		}else
		{
			l += lum[3 * i]; l += lum[3 * i + 1]; l += lum[3 * i + 2];
			l_x += lum_dx[3 * i]; l_x += lum_dx[3 * i + 1]; l_x += lum_dx[3 * i + 2];
			l_y += lum_dy[3 * i]; l_y += lum_dy[3 * i + 1]; l_y += lum_dy[3 * i + 2];
		}

		//Float lum_total = (alpha*l + l_x + l_y)*(alpha*l + l_x + l_y) + epsilon;
		Float lum_total = l*l + epsilon;

		//read primal variance (multiplied by alpha and normalize by luminance)
		Float e = prim_var[3 * i];
		e += prim_var[3 * i + 1];
		e += prim_var[3 * i + 2];
		e *= alpha*alpha;

		//read gradients' variance and normalize it
		Float ex = dx_var[3 * i];
		ex += dx_var[3 * i + 1];
		ex += dx_var[3 * i + 2];

		Float ey = dy_var[3 * i];
		ey += dy_var[3 * i + 1];
		ey += dy_var[3 * i + 2];

		if (m_only_gradients) {
			e = ex + ey;	//ignore var(I) !
		}

		if (e < 0.f  || !std::isfinite(e)){
			int y = i / m_size.x;
			int x = i - y*m_size.x;
			Log(EWarn, "\nALERT!!! Sample (%i, %i) has E value that is negative ex=%f, ey=%f!", x, y, ex, ey);
		}

		if (lum_total < 0.f  || !std::isfinite(lum_total)){
			int y = i / m_size.x;
			int x = i - y*m_size.x;
			Log(EWarn, "\nALERT!!! Sample (%i, %i) has LUM value that is negative lum=%f!", x, y, lum_total);
		}

		//caution here, for large spp underflow is possible!
		double weight =  std::max(double(0.f), std::min(double(1.f), 
							((double(e) / double(m_accum_map[i] * m_accum_map[i])					// normalization variance for samples (n^2)
							/double(lum_total))														// relative variance
							* (double(m_requested_spp) / double(m_requested_spp + m_accum_map[i])))	// see Fabrice's adaptive sampling paper, that is the RMSE reduction if we add m_requested_spp samples
							/double(m_accum_map[i])));												// normalization for variance of mean (n)
	
		m_tmp[i] = weight;

		if (m_tmp[i] < 0.f || !std::isfinite(m_tmp[i])){
			int y = i / m_size.x;
			int x = i - y*m_size.x;
			Log(EWarn, "\nALERT!!! Sample (%i, %i) before processing has value %f!", x, y, m_tmp[i]);
			m_tmp[i] = Float(0.f);
		}



	}

	//process data with median filter followed by max filter
	processEnergy(2, 0);

	for (int i = 0; i < npix; i++){
		energy += m_tmp[i];
		if (m_tmp[i] < 0.f || !std::isfinite(m_tmp[i])){
			int y = i / m_size.x;
			int x = i - y*m_size.x;
			Log(EWarn, "\nALERT!!! Sample (%i, %i) after median has value %f!", x, y, m_tmp[i]);
			m_tmp[i] = Float(0.f);
		}
	}

	if (energy < 0.f ||  !std::isfinite(energy)){
		Log(EWarn, "\nALERT!!! Negative energy, we are lost! value %f!", energy);
	}


	Float weights[] ={ 0.1, 0.2, 0.4, 0.2, 0.1 };

	int len = 2;
	splatInSPP(energy, weights, len);

	//compute average spp and max spp
	Float min_spp = 10000000;
	Float max_spp = 0;
	Float mean_spp = 0;
	for (int i = 0; i < npix; i++){
		if (m_tmp[i] < 0.f || !std::isfinite(m_tmp[i])){
			int y = i / m_size.x;
			int x = i - y*m_size.x;
			Log(EWarn, "\nALERT!!! Sample (%i, %i) after splatting has value %f!", x, y, m_tmp[i]);
			m_tmp[i] = Float(0.f);
		}
		mean_spp += m_spp[i];
		max_spp = std::max(max_spp, Float(m_spp[i]));
		min_spp = std::min(min_spp, Float(m_spp[i]));
	}
	mean_spp /= npix;

	Log(EInfo, "\nIntermediate sampling map stats: \nspp(mean): %.6f, spp(max): %.6f, spp(min): %.6f", mean_spp, max_spp, min_spp);
	Assert(mean_spp <= m_requested_spp + 1.0 && max_spp <= m_maxspp + 1.0); //+1 to account for rounding and accumulation errors

	Float old_mean = mean_spp;

	//because of clamping to m_maxspp we usually have "mean_spp < m_requested_spp" => Redistribute remaining samples uniformly!
	//note that we don't check if a pixel is already saturated, since free_spp is usually very small and a few more samples won't hurt much in saturated samples
	Float free_spp = std::max(/*Float(0.0)*/Float(2.0), Float(m_requested_spp) - mean_spp);
	mean_spp = redistributeUniformly(free_spp);

	//get some more statistics
	int max_map = -1000000;
	int min_map = 1000000;
	for (int i = 0; i < npix; i++){
		min_map = std::min(min_map, m_map[i]);
		max_map = std::max(max_map, m_map[i]);
	}

	Log(EInfo, "\nGenerated sampling map: Requested: \n %i; max allowed: %i", m_requested_spp, m_maxspp);
	Log(EInfo, "\nFinished generating sampling map: spp(total): %.6f \ndetails: spp(distr.): %.6f, spp(redistr.): %.6f, min: %i, max: %i", mean_spp, old_mean, free_spp, min_map, max_map);
	Assert(min_map >= 0 /*&& max_map < m_maxspp + free_spp + 1*/);

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////!!!/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void SamplingMap::processEnergy(int rad_median, int rad_max)
{

	int npix = m_size.x * m_size.y, el, offset;

	//median on m_tmp -> m_tmp2 
	int nn = std::pow(2 * rad_median + 1, 2);
	std::vector<Float> neighbourhood(nn);
	for (int i=0; i < npix; i++){
		el=0;
		for (int x=-rad_median; x <= rad_median; x++){
			for (int y=-rad_median; y <= rad_median; y++,el++){
				offset = i + y*m_size.x + x;
				neighbourhood[el] = (offset >= 0 && offset<npix) ? m_tmp[offset] : 0.f;
			}
		}
		std::sort(neighbourhood.begin(), neighbourhood.end());
		m_tmp2[i] = neighbourhood[int(nn / 2)];
	}

	//max on m_tmp2 -> m_tmp
	if (rad_max > 0){
		int nn2 = std::pow(2 * rad_max + 1, 2);
		std::vector<Float> neighbourhood2(nn2);
		for (int i=0; i < npix; i++){
			el=0;
			for (int x=-rad_max; x <= rad_max; x++){
				for (int y=-rad_max; y <= rad_max; y++, el++){
					offset = i + y*m_size.x + x;
					neighbourhood2[el] = (offset >= 0 && offset < npix) ? m_tmp2[offset] : 0.f;
				}
			}
			std::sort(neighbourhood2.begin(), neighbourhood2.end());
			m_tmp[i] = neighbourhood2[nn2 - 1];
		}
	}
	else{
		for (int i=0; i < npix; i++){
			m_tmp[i] = m_tmp2[i];
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Float SamplingMap::redistributeUniformly(const Float &free_spp)
{
	Float mean_spp = Float(0.0);
	int npix = m_size.x * m_size.y;

	for (int i = 0; i < npix; i++){
		int spp_int = int(free_spp + m_spp[i]);
		Float residual = free_spp + m_spp[i] - spp_int;

		m_map[i] = (m_rgn->nextFloat() < residual) ? spp_int + 1 : spp_int;

		mean_spp += m_map[i];
	//	m_accum_map[i] += m_map[i]; //add samples to accumulated map
	}
	return mean_spp / npix;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Float  SamplingMap::splatInSPP(const Float &total, Float *weights, int len)
{
	int min_spp = 2;
	Assert(m_requested_spp >= 2);

	//check weights and normalize if necessary
	Float accum_w = Float(0.0);
	for (int i = 0; i < 2 * len + 1; i++)
		accum_w += weights[i];
	Float c = Float(1.0) / accum_w;

	int npix = m_size.x * m_size.y;

	Float spp, spp_o;
	int x, y;
	for (int i = 0; i < npix; i++){
		//we distribute spp relative to total. We distribute only requested_Spp-2 samples to ensure that at least two samples always distributed in a pixel
		spp = (m_requested_spp - min_spp) * npix * (m_tmp[i] / total);	

		y = i / m_size.x;
		x = i - y*m_size.x;

		if (y > len - 1 && y < m_size.y - len && x > len - 1 && x < m_size.x - len){
			for (int xo = -len; xo <= len; xo++)
				for (int yo = -len; yo <= len; yo++){
					int idx_o = i + yo*m_size.x + xo;

					Assert(idx_o >= 0 && idx_o < npix);

					spp_o = m_spp[idx_o] + c*weights[xo + len] * c*weights[yo + len] * spp;
					m_spp[idx_o] = std::min(spp_o, Float(m_maxspp));
				}
		}
		else{
			spp_o = m_spp[i] + spp;
			m_spp[i] = std::min(spp_o, Float(m_maxspp));
		}
	}

	return Float(0.0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int SamplingMap::get(int x, int y)
{
	if (x>=0 && x < m_size.x && y>=0 && y < m_size.y)
		return m_map[y*m_size.x + x];
	return m_requested_spp;
}

int SamplingMap::getAccum(int x, int y)
{
	if (x >= 0 && x < m_size.x && y >= 0 && y < m_size.y)
		return m_accum_map[y*m_size.x + x];
	return m_requested_spp;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SamplingMap::getImageAccum(Bitmap *bitmap)
{
	int npix = m_size.x * m_size.y;
	int x, y;
	for (int i = 0; i < npix; i++){
		y = i / m_size.x;
		x = i - y*m_size.x;
		bitmap->setPixel(Point2i(x, y), Spectrum(m_accum_map[i]));
	}
}

void SamplingMap::getImage(Bitmap *bitmap)
{
	int npix = m_size.x * m_size.y;
	int x, y;
	for (int i = 0; i < npix; i++){
		y = i / m_size.x;
		x = i - y*m_size.x;
		bitmap->setPixel(Point2i(x, y), Spectrum(m_map[i]));
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

SamplingMap::SamplingMap(Stream *stream, InstanceManager *manager)
{
	m_rgn = static_cast<Random *>(manager->getInstance(stream));
	m_size = Vector2i(stream);
	m_maxspp = stream->readInt();
	m_requested_spp = stream->readInt();
	//m_real_spp = stream->readInt();
	int npix = m_size.x * m_size.y;
	for (int i = 0; i < npix; i++){
		m_map[i] = stream->readInt();
		m_accum_map[i] = stream->readInt();
		m_tmp[i] = stream->readFloat();
		m_tmp2[i] = stream->readFloat();
		m_spp[i] = stream->readFloat();
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SamplingMap::serialize(Stream *stream, InstanceManager *manager) const
{
	manager->serialize(stream, m_rgn.get());
	m_size.serialize(stream);
	stream->writeInt(m_maxspp);
	stream->writeInt(m_requested_spp);
//	stream->writeInt(m_real_spp);
	int npix = m_size.x * m_size.y;
	for (int i = 0; i < npix; i++){
		stream->writeInt(m_map[i]);
		stream->writeInt(m_accum_map[i]);
		stream->writeFloat(m_tmp[i]);
		stream->writeFloat(m_tmp2[i]);
		stream->writeFloat(m_spp[i]);
	}
}
MTS_NAMESPACE_END
