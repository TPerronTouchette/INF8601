#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "log.h"
#include "sinoscope.h"


typedef struct sinoscope_params {
    float interval_inverse;
    float time;
    float max;
    float phase0;
    float phase1;
    float dx;
    float dy;

    unsigned int width;
    unsigned int height;
    unsigned int taylor;
    unsigned int interval;
} sinoscope_params_t;


int sinoscope_opencl_init(sinoscope_opencl_t* opencl, cl_device_id opencl_device_id, unsigned int width,
			  unsigned int height) {

	cl_int ret;

	// Setting device ID
	opencl->device_id = opencl_device_id;

	// Create the compute context
	opencl->context = clCreateContext(NULL, 1, &opencl_device_id,NULL,NULL,&ret);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clCreateContext failed (%d)", ret);
		goto fail_exit;
	}

	// Create a command queue
	opencl->queue = clCreateCommandQueue(opencl->context, opencl_device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ,&ret);
	if (ret != CL_SUCCESS){
		LOG_ERROR("clCreateCommandQueue failed (%d)", ret);
		goto fail_exit;
	}
	int opencl_load_kernel_code(char** code, size_t* len);
	
	// Create the compute program from the source buffer
	char* sinoscope_code = NULL;
	size_t sinoscope_code_len = 0;
	opencl_load_kernel_code(&sinoscope_code, &sinoscope_code_len);
	printf("This is the kernel code length: %zu\n", sinoscope_code_len);
	printf("This is the code I think: %s\n", sinoscope_code);
	cl_program program = clCreateProgramWithSource(opencl->context, 1, (const char**)&sinoscope_code,
						 &sinoscope_code_len, &ret);
	if (ret != CL_SUCCESS){
		LOG_ERROR("clCreateProgramWithSource failed (%d)", ret);
		goto fail_exit;
	}
		
	// Create the arrays in shared memory for calculation
	size_t size = 3 * width * height;
	opencl->buffer = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE,size,NULL,&ret); // host_ptr? // Could also be a clCreateImage2D
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clCreateBuffer failed (%d)", ret);
		goto fail_exit;
	}

	// Build the program executable
	cl_device_id devices[] = {opencl_device_id};
	ret = clBuildProgram(program,1,devices,NULL,NULL,NULL);
	if (ret != CL_SUCCESS){
		LOG_ERROR("clBuildProgram failed (%d)", ret);
		goto fail_exit;
	}

	// Create the compute kernel in the program we wish to run
	opencl->kernel = clCreateKernel(program, "kernel_sinoscope", &ret);
	if (ret != CL_SUCCESS){
		LOG_ERROR("clCreateKernel failed (%d)", ret);
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

	sinoscope_params_t* params = malloc(sizeof(sinoscope_params_t));
	// float
	params->interval_inverse = sinoscope->interval_inverse;
	params->time = sinoscope->time;
	params->max = sinoscope->max;
	params->phase0 = sinoscope->phase0;
	params->phase1 = sinoscope->phase1;
	params->dx = sinoscope->dx;
	params->dy = sinoscope->dy;

	//uint
	params->width = sinoscope->width;
	params->height = sinoscope->height;
	params->taylor = sinoscope->taylor;
	params->interval = sinoscope->interval;



	// Exécution du kernel OpenCl
	cl_int ret;
	ret = clSetKernelArg(sinoscope->opencl->kernel, 0, sizeof(cl_mem),  (void*)&sinoscope->opencl->buffer);
	if (ret != CL_SUCCESS){
		LOG_ERROR("clSetKernelArg 0 failed (%d)", ret);
		goto fail_exit;
	}
	ret = clSetKernelArg(sinoscope->opencl->kernel, 1, sizeof(sinoscope_params_t),  (void*)&params);
	if (ret != CL_SUCCESS){
		LOG_ERROR("clSetKernelArg 1 failed (%d)", ret);
		goto fail_exit;
	}

	// Execute the kernel over the entire range
	size_t global_work_size[] = { sinoscope->buffer_size };
	size_t local_work_size[] = { sinoscope->buffer }; 

	ret = clEnqueueNDRangeKernel(sinoscope->opencl->queue, sinoscope->opencl->kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	
	if (ret != CL_SUCCESS){
		LOG_ERROR("clEnqueueNDRangeKernel failed (%d)", ret);
		goto fail_exit;
	}


	return 0;

fail_exit:
    return -1;
}
