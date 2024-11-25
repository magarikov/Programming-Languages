#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#define N 4


const char* kernel_code =
"__kernel void booba(__global float* matrix, __global float* matrix_result, int N, int step) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"

    // Если текущая строка ведущая, просто копируем элементы
"    if (row == step) {\n"
"        matrix_result[row * N + col] = matrix[row * N + col];\n"
"        return;\n"
"    }\n"

    // Обновляем строки ниже ведущей
"    if (row > step && col >= step) {\n"
"        float x = matrix[row * N + step] / matrix[step * N + step];\n"
"        float new_value = matrix[row * N + col] - x * matrix[step * N + col];\n"
"        if (col == step) matrix_result[row * N + col] = 0;\n"
"        else matrix_result[row * N + col] = new_value;\n"
"    }\n"
"}\n";


int main() {
    
    int n = N; 
    float* matrix = (float*)malloc(N * N * sizeof(float));
    float* matrix_result = (float*)malloc(N * N * sizeof(float));
    srand((unsigned int)time(NULL));

    // заполняем матрицу
    printf("Source matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = (rand() % 21) - 10; 
            //matrix_result[i][j] = matrix[i][j];
            printf("%6.2f ", matrix[i * N + j]);
        }
        printf("\n");
    }

    // переменные для openCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context; // контекст объединяет устройства, память и программу - грубо - среда работы
    cl_command_queue queue; // очередь используется для отправки комманд на устройство
    cl_program program;
    cl_kernel kernel;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL); // 1 - количество строк. Но у нас как бы одна строка из-за \n
    kernel = clCreateKernel(program, "booba", NULL);

    // буфферы передают данные между CPU и GPU
    // CL_MEM_READ_WRITE - буфер позволяет как чтение, так и запись с устройства
    // CL_MEM_COPY_HOST_PTR - указывает, что начальные данные должны быть скопированы из памяти хоста в буфер.
    cl_mem buffer_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * N * sizeof(float), matrix, NULL);
    cl_mem buffer_result = clCreateBuffer(context, CL_MEM_READ_WRITE, N * N * sizeof(float), NULL, NULL);


    size_t work_zone[2] = { N, N };
    clSetKernelArg(kernel, 2, sizeof(int), &n);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_matrix);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_result);
    int t1 = clock();
    for (int step = 0; step < N; step++) {
        clSetKernelArg(kernel, 3, sizeof(int), &step);
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, work_zone, NULL, 0, NULL, NULL);
        clFinish(queue);
        clEnqueueCopyBuffer(queue, buffer_result, buffer_matrix, 0, 0, N * N * sizeof(float), 0, NULL, NULL);
    }
    int t2 = clock();
    printf("\n%d\n", t2 - t1);
    // считываем данные из буфера
    clEnqueueReadBuffer(queue, buffer_matrix, CL_TRUE, 0, N * N * sizeof(float), matrix, 0, NULL, NULL);
    


    printf("Result matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", matrix[i * N + j]);
        }
        printf("\n");
    }

    clReleaseMemObject(buffer_matrix);
    clReleaseMemObject(buffer_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    
    return 0;
}
