__kernel void vector_add_gpu(__global const float* src_matrix,
	__global const float* src_kernel,
	__global float* res,
    const unsigned MATRIX_SIZE)
{
    int extended_size = MATRIX_SIZE + KERNEL_SIZE - 1;

    // constant offset for any extended matrix element
    int offset = KERNEL_SIZE / 2 // offset caused by leading zeros in each matrix column 
        + extended_size * (KERNEL_SIZE / 2); // offset caused by first rows being all zeros

    __local int index_relocations[KERNEL_SIZE * KERNEL_SIZE];
    const int local_id = get_local_id(0);
    if (local_id < KERNEL_SIZE * KERNEL_SIZE) {
        index_relocations[local_id] = extended_size * (local_id / KERNEL_SIZE - KERNEL_SIZE / 2) + local_id % KERNEL_SIZE - KERNEL_SIZE/2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int global_id = get_global_id(0);
    if (global_id >= MATRIX_SIZE * MATRIX_SIZE)
        return;

    float result = 0;

    // this is where global_id is really mapped in our extended matrix
    int index = (global_id / MATRIX_SIZE) * extended_size + global_id % MATRIX_SIZE; 
    for (int idx = 0; idx < KERNEL_SIZE * KERNEL_SIZE; idx++) {
        result += src_matrix[offset + index + index_relocations[idx]] * src_kernel[idx];
        
    }
    res[global_id] = result;
}