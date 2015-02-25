#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>
#include <fstream>

#include "timer.h"
#include "utils.h"

using namespace std;


#define SCALE 2.0
#define SHIFT 4.5
#define BLOCKS 1500
#define THREADS 1024
#define ARRAYSIZE 153600000


/* we are generating 1500 * 1024 * 100 = 153 600 000 random numbers on the device and then copy back to the host */

// set up the curandState for each individual thread
__global__ 
void setupSeeds(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence
       number, no offset */
    /* seed , sequence, offset, state */
    curand_init(7+id, id, 0, &state[id]);
}

// the kernel for TEST 1
// the internal looping mode
// each thread would have to loop 100 times
__global__
void test1(curandState *state, float *outputArray){
    int myID = threadIdx.x + blockIdx.x * blockDim.x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[myID];
    /* Generate pseudo-random uniforms */

    for ( int i = 0; i < 100; ++i){
        outputArray[myID * 100 + i] = curand_normal(&localState);
    }

}




// the kernel for TEST 2
// the external looping mode
__global__ 
void test2(curandState *state, float *outputArray)
{
    int myID = threadIdx.x + blockIdx.x * blockDim.x;

    outputArray[myID] = curand_normal(&state[myID]);
}



void demo()
{


    curandState *d_curandStates;
    float *d_array, *h_array;
    int deviceIdx;
    struct cudaDeviceProp deviceProperties;



    checkCudaErrors(cudaGetDevice(&deviceIdx));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProperties,deviceIdx));



    h_array = (float *)calloc(ARRAYSIZE, sizeof(float));

    /* Allocate space for prng states on device */
    // a specific state for each thread
    checkCudaErrors(cudaMalloc((void **)&d_curandStates, THREADS * BLOCKS * sizeof(curandState)));

    ofstream fs;




    GpuTimer timer1, timer2;


    /************** TEST 1 ************/
    /********** INTERNAL LOOPING ******/

    /* Setup prng states */
    setupSeeds<<<BLOCKS, THREADS>>>(d_curandStates);

    checkCudaErrors(cudaMalloc(&d_array, ARRAYSIZE * sizeof(float)));  
    checkCudaErrors(cudaMemset(d_array, 0, ARRAYSIZE * sizeof(float))); 

    timer1.Start();

    test1<<<BLOCKS, THREADS>>>(d_curandStates, d_array);
    checkCudaErrors(cudaMemcpy(h_array, d_array, ARRAYSIZE * sizeof(float), cudaMemcpyDeviceToHost));

    timer1.Stop();

    // fs.open("test1.txt");
    // for ( size_t i = 0; i < ARRAYSIZE; ++i){
    //     fs << h_array[i] << endl;
    // }
    // fs.close();

    checkCudaErrors(cudaFree(d_array));
  



    /************** TEST 2 ************/
    /********** EXTERNEL LOOPING ******/

    // first reallocate the d_array
    checkCudaErrors(cudaMalloc(&d_array, BLOCKS * THREADS * sizeof(float)));
    checkCudaErrors(cudaMemset(d_array, 0, BLOCKS * THREADS * sizeof(float)));
    
    // refresh the curandState
    setupSeeds<<<BLOCKS, THREADS>>>(d_curandStates);



    timer2.Start();

    for (int i = 0; i < 100; ++i){

        test2<<<BLOCKS, THREADS>>>(d_curandStates, d_array);
        checkCudaErrors(cudaMemcpy(h_array + i * 100, d_array, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost));

    }

    timer2.Stop();

    // fs.open("test2.txt");
    // for ( size_t i = 0; i < ARRAYSIZE; ++i){
    //     fs << h_array[i] << endl;
    // }
    // fs.close();

    int err = printf("TEST 1 ran in: %f msecs.\n", timer1.Elapsed());

    if (err < 0) {
      std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
      exit(1);
    }

    err = printf("TEST 2 ran in: %f msecs.\n", timer2.Elapsed());

    if (err < 0) {
      std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
      exit(1);
    }


    /* Cleanup */
    checkCudaErrors(cudaFree(d_curandStates));
    checkCudaErrors(cudaFree(d_array));
    free(h_array);

    return;
}