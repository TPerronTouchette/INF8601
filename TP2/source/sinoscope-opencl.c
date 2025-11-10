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
} sinoscope_params_t;

int sinoscope_opencl_init(sinoscope_opencl_t* opencl, cl_device_id opencl_device_id, unsigned int width,
			  unsigned int height) {

	cl_int ret;

	opencl->device_id = opencl_device_id;

	opencl->context = clCreateContext(NULL, 1, &opencl_device_id, NULL, NULL, &ret);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clCreateContext failed (%d)", ret);
		goto fail_exit;
	}

	opencl->queue = clCreateCommandQueue(opencl->context, opencl_device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clCreateCommandQueue failed (%d)", ret);
		goto fail_exit;
	}

	int opencl_load_kernel_code(char** code, size_t* len);
	char* sinoscope_code = NULL;
	size_t sinoscope_code_len = 0;
	opencl_load_kernel_code(&sinoscope_code, &sinoscope_code_len);

	cl_program program = clCreateProgramWithSource(opencl->context, 1, (const char**)&sinoscope_code,
						       &sinoscope_code_len, &ret);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clCreateProgramWithSource failed (%d)", ret);
		goto fail_exit;
	}

	size_t size = 3 * width * height;
	opencl->buffer = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, size, NULL, &ret);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clCreateBuffer failed (%d)", ret);
		goto fail_exit;
	}

	cl_device_id devices[] = {opencl_device_id};
	ret = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clBuildProgram failed (%d)", ret);
		goto fail_exit;
	}

	opencl->kernel = clCreateKernel(program, "kernel_sinoscope", &ret);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clCreateKernel failed (%d)", ret);
		goto fail_exit;
	}

	return 0;

fail_exit:
	return -1;
}

void sinoscope_opencl_cleanup(sinoscope_opencl_t* opencl)
{
	if (opencl->kernel)
		clReleaseKernel(opencl->kernel);
	if (opencl->queue)
		clReleaseCommandQueue(opencl->queue);
	if (opencl->context)
		clReleaseContext(opencl->context);
	if (opencl->buffer)
		clReleaseMemObject(opencl->buffer);
}

int sinoscope_image_opencl(sinoscope_t* sinoscope) {
	if (!sinoscope) {
		LOG_ERROR_NULL_PTR();
		goto fail_exit;
	}

	cl_int ret;

	sinoscope_params_t params = {
		.interval_inverse = sinoscope->interval_inverse,
		.time = sinoscope->time,
		.max = sinoscope->max,
		.phase0 = sinoscope->phase0,
		.phase1 = sinoscope->phase1,
		.dx = sinoscope->dx,
		.dy = sinoscope->dy
	};

	cl_mem params_buf = clCreateBuffer(
		sinoscope->opencl->context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(sinoscope_params_t),
		&params,
		&ret
	);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clCreateBuffer params failed (%d)", ret);
		goto fail_exit;
	}

	ret  = clSetKernelArg(sinoscope->opencl->kernel, 0, sizeof(cl_mem), &sinoscope->opencl->buffer);
	ret |= clSetKernelArg(sinoscope->opencl->kernel, 1, sizeof(cl_mem), &params_buf);
	ret |= clSetKernelArg(sinoscope->opencl->kernel, 2, sizeof(unsigned int), &sinoscope->width);
	ret |= clSetKernelArg(sinoscope->opencl->kernel, 3, sizeof(unsigned int), &sinoscope->height);
	ret |= clSetKernelArg(sinoscope->opencl->kernel, 4, sizeof(unsigned int), &sinoscope->taylor);
	ret |= clSetKernelArg(sinoscope->opencl->kernel, 5, sizeof(unsigned int), &sinoscope->interval);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clSetKernelArg failed (%d)", ret);
		goto fail_exit;
	}

	size_t global_work_size[2] = { sinoscope->width, sinoscope->height };
	size_t local_work_size[2] = { 32, 32 };

	ret = clEnqueueNDRangeKernel(
		sinoscope->opencl->queue,
		sinoscope->opencl->kernel,
		2,
		NULL,
		global_work_size,
		local_work_size,
		0,
		NULL,
		NULL
	);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clEnqueueNDRangeKernel failed (%d)", ret);
		goto fail_exit;
	}

	ret = clFinish(sinoscope->opencl->queue);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clFinish failed (%d)", ret);
		goto fail_exit;
	}

	ret = clEnqueueReadBuffer(
		sinoscope->opencl->queue,
		sinoscope->opencl->buffer,
		CL_TRUE,
		0,
		3 * sinoscope->width * sinoscope->height,
		sinoscope->buffer,
		0,
		NULL,
		NULL
	);
	if (ret != CL_SUCCESS) {
		LOG_ERROR("clEnqueueReadBuffer failed (%d)", ret);
		goto fail_exit;
	}

	clReleaseMemObject(params_buf);
	return 0;

fail_exit:
	if (params_buf)
		clReleaseMemObject(params_buf);
	return -1;
}
