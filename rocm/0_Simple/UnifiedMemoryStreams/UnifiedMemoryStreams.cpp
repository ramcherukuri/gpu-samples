/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a simple task consumer using threads and streams
 * with all data in Unified Memory, and tasks consumed by both host and device
 */

// system includes
#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#ifdef USE_PTHREADS
#include <pthread.h>
#else
#include <omp.h>
#endif
#include <stdlib.h>

// cuBLAS
#include <hipblas.h>

// utilities
#include <helper_hip.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// SRAND48 and DRAND48 don't exist on windows, but these are the equivalent functions
	void srand48(long seed)
	{
		srand((unsigned int)seed);
	}
	double drand48()
	{
		return double(rand())/RAND_MAX;
	}
#endif

const char *sSDKname = "UnifiedMemoryStreams";

// simple task
template <typename T>
struct Task
{
    unsigned int size, id;
    T *data;
    T *result;
    T *vector;

    Task() : size(0), id(0), data(NULL), result(NULL), vector(NULL) {};
    Task(unsigned int s) : size(s), id(0), data(NULL), result(NULL)
    {
        // allocate unified memory -- the operation performed in this example will be a DGEMV
        checkHIPErrors(hipMallocManaged(&data, sizeof(T)*size*size));
        checkHIPErrors(hipMallocManaged(&result, sizeof(T)*size));
        checkHIPErrors(hipMallocManaged(&vector, sizeof(T)*size));
        checkHIPErrors(hipDeviceSynchronize());
    }

    ~Task()
    {
        // ensure all memory is deallocated
        checkHIPErrors(hipDeviceSynchronize());
        checkHIPErrors(hipFree(data));
        checkHIPErrors(hipFree(result));
        checkHIPErrors(hipFree(vector));
    }

    void allocate(const unsigned int s, const unsigned int unique_id)
    {
        // allocate unified memory outside of constructor
        id = unique_id;
        size = s;
        checkHIPErrors(hipMallocManaged(&data, sizeof(T)*size*size));
        checkHIPErrors(hipMallocManaged(&result, sizeof(T)*size));
        checkHIPErrors(hipMallocManaged(&vector, sizeof(T)*size));
        checkHIPErrors(hipDeviceSynchronize());

        // populate data with random elements
        for (int i=0; i<size*size; i++)
        {
            data[i] = drand48();
        }

        for (int i=0; i<size; i++)
        {
            result[i] = 0.;
            vector[i] = drand48();
        }
    }
};

#ifdef USE_PTHREADS
struct threadData_t
{
    int tid;
    Task<double> *TaskListPtr;
    hipStream_t *streams;
    hipblasHandle_t *handles;
    int taskSize;
};

typedef struct threadData_t threadData;
#endif


// simple host dgemv: assume data is in row-major format and square
template <typename T>
void gemv(int m, int n, T alpha, T *A, T *x, T beta, T *result)
{
    // rows
    for (int i=0; i<n; i++)
    {
        result[i] *= beta;

        for (int j=0; j<n; j++)
        {
            result[i] += A[i*n+ j]*x[j];
        }
    }
}

// execute a single task on either host or device depending on size
#ifdef USE_PTHREADS
void* execute(void* inpArgs)
{
    threadData *dataPtr    = (threadData *) inpArgs;
    hipStream_t *stream   = dataPtr->streams;
    hipblasHandle_t *handle = dataPtr->handles;
    int tid                = dataPtr->tid;

    for (int i = 0; i < dataPtr->taskSize; i++)
    {
        Task<double>  &t           = dataPtr->TaskListPtr[i];

        if (t.size < 100)
        {
            // perform on host
            printf("Task [%d], thread [%d] executing on host (%d)\n",t.id,tid,t.size);

            // attach managed memory to a (dummy) stream to allow host access while the device is running
            checkHIPErrors(cudaStreamAttachMemAsync(stream[0], t.data, 0, hipMemAttachHost));
            checkHIPErrors(cudaStreamAttachMemAsync(stream[0], t.vector, 0, hipMemAttachHost));
            checkHIPErrors(cudaStreamAttachMemAsync(stream[0], t.result, 0, hipMemAttachHost));
            // necessary to ensure Async cudaStreamAttachMemAsync calls have finished
            checkHIPErrors(hipStreamSynchronize(stream[0]));
            // call the host operation
            gemv(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
        }
        else
        {
            // perform on device
            printf("Task [%d], thread [%d] executing on device (%d)\n",t.id,tid,t.size);
            double one = 1.0;
            double zero = 0.0;

            // attach managed memory to my stream
            checkHIPErrors(hipblasSetStream(handle[tid+1], stream[tid+1]));
            checkHIPErrors(cudaStreamAttachMemAsync(stream[tid+1], t.data, 0, cudaMemAttachSingle));
            checkHIPErrors(cudaStreamAttachMemAsync(stream[tid+1], t.vector, 0, cudaMemAttachSingle));
            checkHIPErrors(cudaStreamAttachMemAsync(stream[tid+1], t.result, 0, cudaMemAttachSingle));
            // call the device operation
            checkHIPErrors(hipblasDgemv(handle[tid+1], HIPBLAS_OP_N, t.size, t.size, &one, t.data, t.size, t.vector, 1, &zero, t.result, 1));
        }
    }

    pthread_exit(NULL);
}
#else
template <typename T>
void execute(Task<T> &t, hipblasHandle_t *handle, hipStream_t *stream, int tid)
{
    if (t.size < 100)
    {
        // perform on host
        printf("Task [%d], thread [%d] executing on host (%d)\n",t.id,tid,t.size);

        // attach managed memory to a (dummy) stream to allow host access while the device is running
        checkHIPErrors(cudaStreamAttachMemAsync(stream[0], t.data, 0, hipMemAttachHost));
        checkHIPErrors(cudaStreamAttachMemAsync(stream[0], t.vector, 0, hipMemAttachHost));
        checkHIPErrors(cudaStreamAttachMemAsync(stream[0], t.result, 0, hipMemAttachHost));
        // necessary to ensure Async cudaStreamAttachMemAsync calls have finished
        checkHIPErrors(hipStreamSynchronize(stream[0]));
        // call the host operation
        gemv(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
    }
    else
    {
        // perform on device
        printf("Task [%d], thread [%d] executing on device (%d)\n",t.id,tid,t.size);
        double one = 1.0;
        double zero = 0.0;

        // attach managed memory to my stream
        checkHIPErrors(hipblasSetStream(handle[tid+1], stream[tid+1]));
        checkHIPErrors(cudaStreamAttachMemAsync(stream[tid+1], t.data, 0, cudaMemAttachSingle));
        checkHIPErrors(cudaStreamAttachMemAsync(stream[tid+1], t.vector, 0, cudaMemAttachSingle));
        checkHIPErrors(cudaStreamAttachMemAsync(stream[tid+1], t.result, 0, cudaMemAttachSingle));
        // call the device operation
        checkHIPErrors(hipblasDgemv(handle[tid+1], HIPBLAS_OP_N, t.size, t.size, &one, t.data, t.size, t.vector, 1, &zero, t.result, 1));
    }
}
#endif

// populate a list of tasks with random sizes
template <typename T>
void initialise_tasks(std::vector< Task<T> > &TaskList)
{
    for (unsigned int i=0; i<TaskList.size(); i++)
    {
        // generate random size
        int size;
        size = std::max((int)(drand48()*1000.0), 64);
        TaskList[i].allocate(size, i);
    }
}

int main(int argc, char **argv)
{
    // set device
    hipDeviceProp_t device_prop;
    int dev_id = findHIPDevice(argc, (const char **) argv);
    checkHIPErrors(hipGetDeviceProperties(&device_prop, dev_id));

    if (!device_prop.managedMemory) { 
        // This samples requires being run on a device that supports Unified Memory
        fprintf(stderr, "Unified Memory not supported on this device\n");

        exit(EXIT_WAIVED);
    }

    if (device_prop.computeMode == hipComputeModeProhibited)
    {
        // This sample requires being run with a default or process exclusive mode
        fprintf(stderr, "This sample requires a device in either default or process exclusive mode\n");

        exit(EXIT_WAIVED);
    }

    // randomise task sizes
    int seed = time(NULL);
    srand48(seed);

    // set number of threads
    const int nthreads = 4;

    // number of streams = number of threads
    hipStream_t *streams = new hipStream_t[nthreads+1];
    hipblasHandle_t *handles = new hipblasHandle_t[nthreads+1];

    for (int i=0; i<nthreads+1; i++)
    {
        checkHIPErrors(hipStreamCreate(&streams[i]));
        checkHIPErrors(hipblasCreate(&handles[i]));
    }

    // create list of N tasks
    unsigned int N = 40;
    std::vector<Task<double> > TaskList(N);
    initialise_tasks(TaskList);

    printf("Executing tasks on host / device\n");

    // run through all tasks using threads and streams
#ifdef USE_PTHREADS
    pthread_t threads[nthreads];
    threadData *InputToThreads = new threadData[nthreads];

    for (int i=0; i < nthreads; i++)
    {
        checkHIPErrors(hipSetDevice(dev_id));
        InputToThreads[i].tid         = i;
        InputToThreads[i].streams     = streams;
        InputToThreads[i].handles     = handles;

        if ((TaskList.size() / nthreads) == 0)
        {
            InputToThreads[i].taskSize    = (TaskList.size() / nthreads);
            InputToThreads[i].TaskListPtr = &TaskList[i*(TaskList.size() / nthreads)];
        }
        else
        {
            if (i == nthreads - 1)
            {
                InputToThreads[i].taskSize    = (TaskList.size() / nthreads) + (TaskList.size() % nthreads);
                InputToThreads[i].TaskListPtr = &TaskList[i*(TaskList.size() / nthreads)+ (TaskList.size() % nthreads)];
            }
            else
            {
                InputToThreads[i].taskSize    = (TaskList.size() / nthreads);
                InputToThreads[i].TaskListPtr = &TaskList[i*(TaskList.size() / nthreads)];
            }
        }

        pthread_create(&threads[i], NULL, &execute, &InputToThreads[i]);
    }
    for (int i=0; i < nthreads; i++)
    {
        pthread_join(threads[i], NULL);
    }
#else
    omp_set_num_threads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<TaskList.size(); i++)
    {
        checkHIPErrors(hipSetDevice(dev_id));
        int tid = omp_get_thread_num();
        execute(TaskList[i], handles, streams, tid);
    }
#endif

    hipDeviceSynchronize();

    // Destroy CUDA Streams, cuBlas handles
    for (int i=0; i<nthreads+1; i++)
    {
        hipStreamDestroy(streams[i]);
        hipblasDestroy(handles[i]);
    }

    // Free TaskList
    std::vector< Task<double> >().swap(TaskList);

    printf("All Done!\n");
    exit(EXIT_SUCCESS);
}
