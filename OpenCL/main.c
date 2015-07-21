#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Constants
//
const int ARRAY_SIZE = 1000;
const int g_num_mem_objs = 5;
const int g_num_params = 20;


///
//  parameters
//
// size_t h_m;


// size_t h_k;


// size_t h_T;


// float h_h;


// float h_r;


// float h_c;


// float h_theta;


// float h_s;


// float h_alpha;


// size_t h_maxhold;

// float h_lambda;

// size_t h_min_demand;

// size_t h_max_demand;

// float *h_demand_distribution;


///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;


    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }


    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }


    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float ** mem_pointers)
{
    memObjects[4 + g_num_params] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, h_valuetable, NULL);
    memObjects[1 + g_num_params] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, h_randomtable, NULL);
    // take this as the value table
    memObjects[2 + g_num_params] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);
    memObjects[3 + g_num_params] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * ARRAY_SIZE, NULL, NULL);
    /* initialize all the memobjs */
    /* make all the objects to writable here for simplicity */
    for (int i = 0; i < g_num_params; ++i){
        memObjects[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), mem_pointers[i], NULL)
    }

    /* valid allocation */
    for (int i = 0; i < g_num_params + 4; ++i){
        if (memObjects[i] == NULL){
                std::cerr << "Error creating memory objects." << std::endl;
            return false;
        }
    }

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < g_num_mem_objs; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}


int main(int argc, char** argv)
{
    /* first setting up the opencl environment */

    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[g_num_mem_objs];
    cl_int errNum;

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    /* initialize the kernels */
    // Create OpenCL program from kernels.cl kernel source
    // for simplicity, I'll temporarily put all the kernels in one cl file
    program = CreateProgram(context, device, "kernels.cl");
    if (program == NULL)
    {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Create OpenCL kernel
    // initialize all the kernels here
    kernel &= clCreateKernel(program, "keDeviceTableInit", NULL);
    kernel &= clCreateKernel(program, "keValueTableUpdateWithPolicy", NULL);
    kernel &= clCreateKernel(program, "kePresetValueTable", NULL);

    kernel &= clCreateKernel(program, "decode", NULL);
    kernel &= clCreateKernel(program, "checkStorage", NULL);
    kernel &= clCreateKernel(program, "stateValue", NULL);

    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    /* initialize the memory */
    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel

    float ** mem_pointers;
    mem_pointers = (float*)malloc(g_num_params * sizeof(float*));

    if (!CreateMemObjects(context, memObjects, mem_pointers))
    {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }


    /* send the data to kernel */
    // Set the kernel arguments (result, a, b)
    // you have to pass all the kernel parameters individually!!!!
    // errNum = clSetKernelArg(kernelname, memobjindex, sizeof(cl_mem), &memObjects[0]);

    /* definitely rewrite this part !!!!!!! WTF !!! */
    // errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    // errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    // if (errNum != CL_SUCCESS)
    // {
    //     std::cerr << "Error setting kernel arguments." << std::endl;
    //     Cleanup(context, commandQueue, program, kernel, memObjects);
    //     return 1;
    // }

/**********************************************/

    /* execute the kernels */
    size_t globalWorkSize[1] = { ARRAY_SIZE };
    size_t localWorkSize[1] = { 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, keDeviceTableInit, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    errNum = clEnqueueNDRangeKernel(commandQueue, kePresetValueTable, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue, program, , memObjects);
        return 1;
    }

    errNum = clEnqueueNDRangeKernel(commandQueue, keValueTableUpdateWithPolicy, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

/**********************************************/

    /* transmit the data back */
    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), result,
                                 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Output the result buffer
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);

    return 0;
}


void printResult(float* valueTable, size_t length, string title = "value table"){
    cout << endl << "Now starting to print out the " << title << ": "  << endl;
    // for (size_t i = 0; i < length - 1; ++i){
    //         cout << valueTable[i] << ", ";
    // }
    // cout << valueTable[length - 1] << endl;
    for (size_t i = 0; i < length; ++i){
        cout << " the node : " << i << " has value : " << valueTable[i] << endl;

    }

    return;

}

void wirteToFile( float * valueTable, size_t length, char const * output_file_name, char const * file_format ){
    if ( do_file_exist(output_file_name) ) {
        string user_option;
        cout << "The file\e[38;5;51m " << output_file_name << "\e[m already exists, overwritten? (y/n) : ";
        cin >> user_option;
        if ( user_option == "y")
            remove(output_file_name);
        else {
            cout << "No file written." << endl;
            user_option = "";
            cout << "Print to the screen? (y/n) : ";
            cin >> user_option;
            if ( user_option == "y" )
                printResult( valueTable, length);
            return;
        }
    }

    ofstream ofs;
    ofs.open (output_file_name, std::ofstream::out | std::ofstream::app);
    cout << "fileformat " << file_format << endl;

    if ( !strcmp(file_format, "normal") ){
        for (size_t i = 0; i < length; ++i){
            ofs << " Node index : " << i << "        value key : " << valueTable[i] << endl;
        }
    }

    if ( !strcmp(file_format, "csv" )){
        cout << "in csv" << endl;
        ofs << "index,    key," << endl;
        for (size_t i = 0; i < length; ++i){
            ofs  << i << ",        " << valueTable[i] <<"," << endl;
        }
    }

    ofs.close();
    cout << "Output file written successfully!" << endl;

    return ;
}
