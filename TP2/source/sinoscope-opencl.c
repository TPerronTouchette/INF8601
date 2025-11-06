#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "log.h"
#include "sinoscope.h"

int sinoscope_opencl_init(sinoscope_opencl_t* opencl, cl_device_id opencl_device_id, unsigned int width,
			  unsigned int height) {

	cl_int* ret;

	opencl->context = clCreateContext(NULL, 1, &opencl_device_id,NULL,NULL,ret);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clCreateContext failed (%d)", ret);
		goto fail_exit;
	}

	size_t size = 3 * width * height; // Is this in bytes?? (i think so) on char is one byte so 3 * 1byte * nb pixels
	opencl->buffer = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE,size,NULL,ret); // host_ptr? // Could also be a clCreateImage2D
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clCreateBuffer failed (%d)", ret);
		goto fail_exit;
	}

	opencl->queue = clCreateCommandQueue(opencl->context, opencl_device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ,ret);
	if (ret != CL_SUCCESS){
		LOG_ERROR("clCreateCommandQueue failed (%d)", ret);
		goto fail_exit;
	}
	int opencl_load_kernel_code(char** code, size_t* len);
	
	char* sinoscope_code = NULL;
	size_t sinoscope_code_len = 0;
	opencl_load_kernel_code(&sinoscope_code, &sinoscope_code_len);
	printf("This is the kernel code length: %zu\n", sinoscope_code_len);
	printf("This is the code I think: %s\n", sinoscope_code);
	cl_program program = clCreateProgramWithSource(opencl->context, 1, (const char**)&sinoscope_code,
						 &sinoscope_code_len, ret);
	if (ret != CL_SUCCESS){
		LOG_ERROR("clCreateProgramWithSource failed (%d)", ret);
		goto fail_exit;
	}

	ret = clBuildProgram(program,1,opencl_device_id,NULL,NULL,NULL);
	if (ret != CL_SUCCESS){
		LOG_ERROR("clBuildProgram failed (%d)", ret);
		goto fail_exit;
	}

	return 0;

fail_exit:
	return -1;
}

void sinoscope_opencl_cleanup(sinoscope_opencl_t* opencl)
{
	// Ici, on veut libérer les éléments instantiés dans sinoscope_opencl_init
	// qui ne sont plus utilisés à l'aide des méthodes "Release"

	if (opencl->kernel != NULL) {
		clReleaseKernel(opencl->kernel);
	}

	if (opencl->queue != NULL) {
		clReleaseCommandQueue(opencl->queue);
	}
	
	if (opencl->context != NULL) {
		clReleaseContext(opencl->context);
	}

	if (opencl->buffer != NULL) {
		clReleaseMemObject(opencl->buffer);
	}

}

int sinoscope_image_opencl(sinoscope_t* sinoscope) {
    if (sinoscope == NULL || sinoscope->opencl == NULL) {
        LOG_ERROR_NULL_PTR();
        goto fail_exit;
    }

	

	return 0;

fail_exit:
    return -1;
}
