#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <memory>
#include <cstring>
#include <CL/opencl.h>
#include <vector>
#include <cmath>

using namespace std;

typedef float element_t;

namespace
{
const string default_input_file = "input.txt";
const string default_output_file = "output.txt";
const size_t WORKGROUP_SIZE = 256;

cl_kernel scan_pad_to_pow2;
cl_kernel scan_subarrays;
cl_kernel scan_inc_subarrays;

// OpenCL Environment variables
cl_device_id device;
cl_context context;
cl_command_queue queue;
}


int get_platforms_and_choose_one(std::shared_ptr<cl_platform_id> &platform) {
    cl_uint platforms_number = 0;

    cl_int error = clGetPlatformIDs(10, nullptr, &platforms_number);
    assert(error == CL_SUCCESS && "Error getting platforms number");
    platform.reset(new cl_platform_id[platforms_number]);

    error = clGetPlatformIDs(platforms_number, platform.get(), nullptr);
    assert(error == CL_SUCCESS && "Error getting platform id");
    int used_platform = 0;

    for (cl_int i = 0; i < platforms_number; i++) {
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

vector<element_t> read_input(const string& input_file) {
    ifstream in(input_file);

    if (in.bad()) {
        cerr << "couldn't open input file: " << input_file << endl;
        exit(1);
    }

    cout << "Reading input from " << input_file << endl;
    size_t size = 0;
    in >> size;

    vector<element_t> array(size);
    for (size_t i = 0; i < size; i++)
        in >> array[i];

    return array;
}

void write_result(const vector<element_t> result, const string &output_file) {
    ofstream out(output_file);
    for (size_t i = 1; i < result.size(); i++) {
        out << result[i] << " ";
    }
    cout << "Result is written to " << output_file << "\n";
}


namespace util
{
namespace detail
{
template<class ...T>
void set_kernel_args(cl_kernel kernel, cl_uint arg_id, size_t arg0, T ...args);

template<class T0, class ...T>
typename std::enable_if<!is_same<size_t, T0>::value>::type set_kernel_args(cl_kernel kernel, cl_uint arg_id, T0 arg0, T ...args);

template<class T0>
typename std::enable_if<!is_same<size_t, T0>::value>::type set_kernel_args(cl_kernel kernel, cl_uint arg_id, T0 arg0) {
    cl_int error = clSetKernelArg(kernel, arg_id, sizeof(T0), &arg0);
    assert(error == CL_SUCCESS && "error setting argument");
}

void set_kernel_args(cl_kernel kernel, cl_uint arg_id, size_t arg0) {
    cl_int error = clSetKernelArg(kernel, arg_id, arg0, nullptr);
    assert(error == CL_SUCCESS && "error setting argument");
}

template<class T0, class ...T>
typename std::enable_if<!is_same<size_t, T0>::value>::type set_kernel_args(cl_kernel kernel, cl_uint arg_id, T0 arg0, T ...args) {
    cl_int error = clSetKernelArg(kernel, arg_id, sizeof(T0), &arg0);
    assert(error == CL_SUCCESS && "error setting argument");
    set_kernel_args(kernel, arg_id + 1, args...);
}

template<class ...T>
void set_kernel_args(cl_kernel kernel, cl_uint arg_id, size_t arg0, T ...args) {
    cl_int error = clSetKernelArg(kernel, arg_id, arg0, nullptr);
    assert(error == CL_SUCCESS && "error setting argument");
    set_kernel_args(kernel, arg_id + 1, args...);
}

}

template<class ...T>
void set_kernel_args(cl_kernel kernel, T ...args) {
    detail::set_kernel_args(kernel, 0, args...);
}

cl_program create_program() {
    const char* path = "kernels/scan.cl";

    std::ifstream t(path);
    std::stringstream source;
    source << t.rdbuf();
    string source_str = source.str();
    size_t src_size = source_str.length();
    const char *source_buf = source_str.c_str();

    cl_int error = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(context, 1, &source_buf, &src_size, &error);
    assert(error == CL_SUCCESS && "Error creating cl program source");

    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    show_build_log(program, device);
    assert(error == CL_SUCCESS);

    return program;
}
}

void create_kernels(cl_program program) {
    cl_int error;
    // Extracting the kernel
    scan_pad_to_pow2 = clCreateKernel(program, "scan_pad_to_pow2", &error);
    assert(error == CL_SUCCESS);

    scan_inc_subarrays = clCreateKernel(program, "scan_inc_subarrays", &error);
    assert(error == CL_SUCCESS);

    scan_subarrays = clCreateKernel(program, "scan_subarrays", &error);
    assert(error == CL_SUCCESS);
}

void check_work_group(cl_device_id device) {
    size_t max_workgroup_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, 0);

    assert(WORKGROUP_SIZE <= max_workgroup_size);
}

void recursive_scan(cl_mem d_data, int n) {
    int k = (int) ceil((float)n/(float)WORKGROUP_SIZE);

    size_t buf_size = sizeof(element_t) * WORKGROUP_SIZE * 2;
    cl_int error;
    if (k == 1) {
        util::set_kernel_args(scan_pad_to_pow2, d_data, buf_size, n);
        error = clEnqueueNDRangeKernel(queue, scan_pad_to_pow2, 1, nullptr,
                                              &WORKGROUP_SIZE, &WORKGROUP_SIZE, 0, NULL, NULL);
        assert(error == CL_SUCCESS);
    } else {
        size_t global_worksize = k * WORKGROUP_SIZE;

        cl_mem d_partial = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(element_t)*k, nullptr, &error);
        util::set_kernel_args(scan_subarrays, d_data, buf_size, d_partial, n);
        error = clEnqueueNDRangeKernel(queue, scan_subarrays, 1, nullptr,
                                       &global_worksize, &WORKGROUP_SIZE, 0, NULL, NULL);
        assert(error == CL_SUCCESS);

        recursive_scan(d_partial, k);
        util::set_kernel_args(scan_inc_subarrays, d_data, buf_size, d_partial, n);
        error = clEnqueueNDRangeKernel(queue, scan_inc_subarrays, 1, nullptr,
                                       &global_worksize, &WORKGROUP_SIZE, 0, NULL, NULL);
        assert(error == CL_SUCCESS);

        clReleaseMemObject(d_partial);
    }
}

void cleanup() {
    clReleaseKernel(scan_inc_subarrays);
    clReleaseKernel(scan_subarrays);
    clReleaseKernel(scan_pad_to_pow2);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

// evaluates inclusive scan
vector<element_t> calc_on_gpu(vector<element_t> array) {
    if (array.empty()) return vector<element_t>{};

    std::shared_ptr<cl_platform_id> platform;
    // Platform

    cl_int used_platform = get_platforms_and_choose_one(platform);

    // Device
    cl_int error = clGetDeviceIDs(platform.get()[used_platform], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(error == CL_SUCCESS && "Error getting device ids");

    // check that we can have work group of 256 items
    check_work_group(device);
    // Context
    context = clCreateContext(0, 1, &device, NULL, NULL, &error);
    assert(error == CL_SUCCESS && "Error creating context");

    // Command-queue
    queue = clCreateCommandQueue(context, device, 0, &error);
    assert(error == CL_SUCCESS && "Error creating command queue");

    cl_program program = util::create_program();
    create_kernels(program);

    cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   array.size() * sizeof(element_t), array.data(), &error);
    assert(error == CL_SUCCESS);

    recursive_scan(d_data, (int) array.size());

    vector<element_t> result(array.size());
    error = clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0, array.size() * sizeof(element_t), result.data(), 0, NULL, NULL);
    assert(error == CL_SUCCESS);

    cleanup();
    clReleaseMemObject(d_data);

    result.push_back(result.back() + array.back());
    return result;
}

namespace {
    vector<element_t> calc_on_cpu(const vector<element_t> input) {
        vector<element_t> result(input.size() + 1);

        for (size_t i = 1; i < result.size() + 1; i++) {
            result[i] = result[i - 1] + input[i - 1];
        }

        return result;
    }
}

int main(int argc, char**argv) {
    const string input_file = argc == 1 ? default_input_file : argv[1];
    const string output_file = argc == 1 ? default_output_file : argv[2];

    vector<element_t> input = read_input(input_file);
    vector<element_t> gpu_result = calc_on_gpu(input);

#define DEBUG_PRINTS
#ifdef DEBUG_PRINTS
    vector<element_t> cpu_result = calc_on_cpu(input);
    bool passed = true;
    for (size_t i = 0; i < input.size() + 1; i++) {
        if (abs(cpu_result[i] - gpu_result[i]) > 0.000001) {
            cerr << "Too big difference in " << i << " position: gpu_result is "
             << gpu_result[i] << ", cpu_result is " << cpu_result[i] << endl;
            passed = false;
        }
    }

    if (passed) {
        cout << "CPU and GPU results are same\n";
    } else {
        cout << "CPU and GPU results differ\n";
    }
#endif

    write_result(gpu_result, output_file);
    return 0;
}