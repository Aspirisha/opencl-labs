#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <memory>
#include <CL/opencl.h>

using namespace std;

void vector_add_cpu(const float* const src_a,
    const float* const src_b,
    float* const res,
    const int size)
{
    for (int i = 0; i < size; i++) {
        res[i] = src_a[i] + src_b[i];
    }
}

int get_platforms_and_choose_one(std::shared_ptr<cl_platform_id> &platform) {
    cl_int error = 0;
    cl_uint platforms_number = 0;

    error = clGetPlatformIDs(10, nullptr, &platforms_number);
    platform.reset(new cl_platform_id[platforms_number]);

    error = clGetPlatformIDs(platforms_number, platform.get(), nullptr);
    assert(error == CL_SUCCESS && "Error getting platform id");
    int used_platform = 0;

    for (cl_int i = 0; i < 2; i++) {
        char pform_vendor[40];
        clGetPlatformInfo(platform.get()[i], CL_PLATFORM_VENDOR, sizeof(pform_vendor),
            &pform_vendor, NULL);

        if (strstr(pform_vendor, "NVIDIA")) {
            used_platform = i;
        }
    }
    return used_platform;
}

void show_build_log(cl_program program, cl_device_id device) {
    // Shows the log
    char* build_log;
    size_t log_size;
    // First call to know the proper size
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    build_log = new char[log_size + 1];

    // Second call to get the log
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    string build_str{ build_log };
    if (string{ build_log }.find_first_not_of("\t\n ") != build_str.npos) {
        cout << build_log << endl;
    }
    delete[] build_log;
}

int main(void) {
    const int MAX_PLATFORMS = 3;
    cout << "Welcome to matrix convolutor\n";

    // OpenCL Environment variables
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int error = CL_SUCCESS;
    cl_uint platforms_number = 0;
    std::shared_ptr<cl_platform_id> platform;
    // Platform
    
    cl_int used_platform = get_platforms_and_choose_one(platform);

    // Device
    error = clGetDeviceIDs(platform.get()[used_platform], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(error == CL_SUCCESS && "Error getting device ids");
    // Context
    context = clCreateContext(0, 1, &device, NULL, NULL, &error);
    assert(error == CL_SUCCESS && "Error creating context");

    // Command-queue
    queue = clCreateCommandQueue(context, device, 0, &error);
    assert(error == CL_SUCCESS && "Error creating command queue");

    const unsigned size = 12345678;
    float* src_a_h = new float[size];
    float* src_b_h = new float[size];
    float* res_h = new float[size];
    // Initialize both vectors
    for (int i = 0; i < size; i++) {
        src_a_h[i] = src_b_h[i] = (float)i;
    }

    const int mem_size = sizeof(float)*size;
    // Allocates a buffer of size mem_size and copies mem_size bytes from src_a_h
    cl_mem src_a_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, src_a_h, &error);
    cl_mem src_b_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, src_b_h, &error);
    cl_mem res_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size, NULL, &error);
    assert(error == CL_SUCCESS);

    
    const char* path = "test.cl";

    std::ifstream t(path);
    std::stringstream source;
    source << t.rdbuf();
    string source_str = source.str();
    size_t src_size = source_str.length();
    const char *source_buf = source_str.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &source_buf, &src_size, &error);
    assert(error == CL_SUCCESS && "Error creating cl program source");

    // Builds the program
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    show_build_log(program, device);
    assert(error == CL_SUCCESS);


    // Extracting the kernel
    cl_kernel vector_add_kernel = clCreateKernel(program, "vector_add_gpu", &error);
    assert(error == CL_SUCCESS);

    // Enqueuing parameters
    // Note that we inform the size of the cl_mem object, not the size of the memory pointed by it
    error = clSetKernelArg(vector_add_kernel, 0, sizeof(cl_mem), &src_a_d);
    error |= clSetKernelArg(vector_add_kernel, 1, sizeof(cl_mem), &src_b_d);
    error |= clSetKernelArg(vector_add_kernel, 2, sizeof(cl_mem), &res_d);
    error |= clSetKernelArg(vector_add_kernel, 3, sizeof(unsigned), &size);
    assert(error == CL_SUCCESS);

    const size_t local_ws = 512;	// Number of work-items per work-group

    cl_ulong local_mem_size;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, 0);
    cout << "Local memory size = " << local_mem_size << endl;

    size_t global_ws = (size / local_ws) * local_ws;
    global_ws += local_ws * (size % local_ws != 0);
    size_t max_workgroup_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof size_t, &max_workgroup_size, 0);
    cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE = " << max_workgroup_size << endl;
    error = clEnqueueNDRangeKernel(queue, vector_add_kernel, 2, NULL, &global_ws, &local_ws, 0, NULL, NULL);
    assert(error == CL_SUCCESS);

    // Reading back
    float* cpu_result = new float[size];
    vector_add_cpu(src_a_h, src_b_h, cpu_result, size);

    float* check = new float[size];
    clEnqueueReadBuffer(queue, res_d, CL_TRUE, 0, mem_size, check, 0, NULL, NULL);

    for (int i = 0; i < size; i++) {
        assert(check[i] == cpu_result[i]);
    }
    // Cleaning up
    delete[] src_a_h;
    delete[] src_b_h;
    delete[] res_h;
    delete[] check;
    delete[] cpu_result;
    clReleaseKernel(vector_add_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseMemObject(src_a_d);
    clReleaseMemObject(src_b_d);
    clReleaseMemObject(res_d);
    return 0;
}