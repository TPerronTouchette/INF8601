#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "log.h"
#include "sinoscope.h"

int sinoscope_opencl_init(sinoscope_opencl_t* opencl, cl_device_id opencl_device_id, unsigned int width,
			  unsigned int height) {
	return -1;
}

void sinoscope_opencl_cleanup(sinoscope_opencl_t* opencl)
{

}

int sinoscope_image_opencl(sinoscope_t* sinoscope) {
	return -1;
}
