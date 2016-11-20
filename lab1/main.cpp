#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <memory>
#include <CL/opencl.h>

using namespace std;

static const string input_file = "input.txt";
static const string output_file = "output.txt";
static const size_t MAX_MATRIX_SIZE = 1024;
static const size_t MAX_KERNEL_SZIE = 9;

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

// reads matrix to be convolved into extended matrix
// for example, if we want to convolve 
// 1 1 1
// 1 1 1
// 1 1 1
// with 3x3 kernel, then extended matrix will look like
// 0 0 0 0 0 
// 0 1 1 1 0
// 0 1 1 1 0
// 0 1 1 1 0
// 0 0 0 0 0 
// this allows avoid extra branching on gpu
void read_input(std::shared_ptr<float> &matrix, std::shared_ptr<float> &kernel, size_t &matrix_size, size_t &kernel_size) {
    ifstream in(input_file);
    in >> matrix_size >> kernel_size;

    if (matrix_size == 0 || matrix_size > MAX_MATRIX_SIZE) {
        throw logic_error{ "Unexpected size of matrix" };
    }

    if (kernel_size == 0 || kernel_size % 2 == 0 || kernel_size > MAX_KERNEL_SZIE) {
        throw logic_error{ "Unexpected size of kernel matrix" };
    }

    size_t extended_size = (kernel_size + matrix_size - 1);
    matrix.reset(new float[extended_size * extended_size]);
    kernel.reset(new float[kernel_size * kernel_size]);
    memset(matrix.get(), 0, sizeof(float) * extended_size * extended_size);
    int matrix_offset = (kernel_size / 2) * extended_size + kernel_size / 2;
    for (size_t i = 0; i < matrix_size; i++) {
        for (size_t j = 0; j < matrix_size; j++) {
            in >> matrix.get()[i * extended_size + matrix_offset + j];
        }
    }
    for (size_t i = 0; i < kernel_size * kernel_size; i++) {
        in >> kernel.get()[i];
    }
}

void write_result(const std::shared_ptr<const float> matrix, size_t matrix_size) {
    ofstream out(output_file);

    const float *mat = matrix.get();
    for (size_t i = 0, idx = 0; i < matrix_size; i++) {
        for (size_t j = 0; j < matrix_size; j++, idx++) {
            out << mat[idx] << " ";
        }
        out << endl;
    }
}


// debug functions
namespace {
    void print_matrix(const string& name, shared_ptr<float> matrix, size_t matrix_size) {
        cout << "===================\n";
        cout << name << "\n";
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                cout << matrix.get()[i * matrix_size + j] << " ";
            }
            cout << endl;
        }
        cout << "===================\n";
    }
}

std::shared_ptr<float> calc_on_cpu(std::shared_ptr<const float> matrix, 
    const std::shared_ptr<float> kernel, int matrix_size, int kernel_size) {

    std::shared_ptr<float> res{ new float[matrix_size * matrix_size] };
    int extended_size = matrix_size + kernel_size - 1;
    int matrix_offset = extended_size * (kernel_size / 2) + kernel_size / 2;
    
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            float result = 0;
            for (int ii = -kernel_size / 2, idx = 0; ii <= kernel_size / 2; ii++) {
                for (int jj = -kernel_size / 2; jj <= kernel_size / 2; jj++, idx++) {
                    result += matrix.get()[matrix_offset + (i + ii) * extended_size + j + jj]
                        * kernel.get()[idx];
                }
            }
            
            res.get()[i * matrix_size + j] = result;
        }
    }

    return res;
}

std::shared_ptr<float> calc_on_gpu(const std::shared_ptr<float> matrix,
    const std::shared_ptr<float> kernel, size_t matrix_size, size_t kernel_size) {
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

    const size_t matrix_elements_number = matrix_size * matrix_size;
    const size_t kernel_elements_number = kernel_size * kernel_size;
    const size_t extended_matrix_elements = (matrix_size + kernel_size - 1) * (matrix_size + kernel_size - 1);
    std::shared_ptr<float> result{ new float[matrix_elements_number] };

    // Allocates a buffer of size mem_size and copies mem_size bytes from src_a_h
    cl_mem src_matrix_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        extended_matrix_elements * sizeof(float), matrix.get(), &error);
    cl_mem src_kernel_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        kernel_elements_number * sizeof(float), kernel.get(), &error);
    cl_mem result_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        matrix_elements_number * sizeof(float), NULL, &error);
    assert(error == CL_SUCCESS);

    const char* path = "convolute.cl";

    std::ifstream t(path);
    std::stringstream source;
    source << t.rdbuf();
    string source_str = source.str();
    size_t src_size = source_str.length();
    const char *source_buf = source_str.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &source_buf, &src_size, &error);
    assert(error == CL_SUCCESS && "Error creating cl program source");

    // Builds the program
    stringstream options_builder;
    options_builder << "-D KERNEL_SIZE=" << kernel_size;
    string options = options_builder.str();
    error = clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL);
    show_build_log(program, device);
    assert(error == CL_SUCCESS);


    // Extracting the kernel
    cl_kernel vector_add_kernel = clCreateKernel(program, "vector_add_gpu", &error);
    assert(error == CL_SUCCESS);

    // Enqueuing parameters
    // Note that we inform the size of the cl_mem object, not the size of the memory pointed by it
    error = clSetKernelArg(vector_add_kernel, 0, sizeof(cl_mem), &src_matrix_d);
    error |= clSetKernelArg(vector_add_kernel, 1, sizeof(cl_mem), &src_kernel_d);
    error |= clSetKernelArg(vector_add_kernel, 2, sizeof(cl_mem), &result_d);
    error |= clSetKernelArg(vector_add_kernel, 3, sizeof(unsigned), &matrix_size);
    assert(error == CL_SUCCESS);

    size_t max_workgroup_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof size_t, &max_workgroup_size, 0);
    const size_t local_ws = max_workgroup_size;	// let it be so

    size_t global_ws = (matrix_elements_number / local_ws) * local_ws;
    global_ws += local_ws * (matrix_elements_number % local_ws != 0);

    error = clEnqueueNDRangeKernel(queue, vector_add_kernel, 1, nullptr, &global_ws, &local_ws, 0, NULL, NULL);
    assert(error == CL_SUCCESS);

    clEnqueueReadBuffer(queue, result_d, CL_TRUE, 0, matrix_elements_number * sizeof(float), result.get(), 0, NULL, NULL);

    // Cleaning up
    clReleaseKernel(vector_add_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseMemObject(src_matrix_d);
    clReleaseMemObject(src_kernel_d);
    clReleaseMemObject(result_d);

    return result;
}

int main(void) {
    const int MAX_PLATFORMS = 3;
    cout << "Welcome to matrix convolutor\n";
    std::shared_ptr<float> matrix;
    std::shared_ptr<float> kernel;
    size_t matrix_size;
    size_t kernel_size;
    read_input(matrix, kernel, matrix_size, kernel_size);
    cout << "Succesfully read data from " << input_file << "\n";
    auto gpu_result = calc_on_gpu(matrix, kernel, matrix_size, kernel_size);
    //#define DEBUG_PRINTS
#ifdef DEBUG_PRINTS
    auto cpu_result = calc_on_cpu(matrix, kernel, matrix_size, kernel_size);
    print_matrix("extended matrix: ", matrix, matrix_size + kernel_size - 1);
    print_matrix("cpu result: ", cpu_result, matrix_size);
    print_matrix("gpu result: ", gpu_result, matrix_size);
    float max_allowed_error = 0.001;
    for (size_t i = 0; i < matrix_size * matrix_size; i++) {
        if (abs(cpu_result.get()[i] - gpu_result.get()[i]) > max_allowed_error) {
            cout << "Difference greater than " << max_allowed_error << " " << cpu_result.get()[i] << " " << gpu_result.get()[i] << endl;
        }
    }
#endif

    write_result(gpu_result, matrix_size);
    cout << "Result is written to " << output_file << "\n";
    return 0;
}