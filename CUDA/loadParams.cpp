#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "parameters.h"
#include "loadParams.h"




#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <iostream>
#include "helper_cuda.h"
#include "helper_math.h"

using namespace std;
using namespace rapidjson;


int loadParams(string filename){

	const char * fname = filename.c_str();
	char  faction = 'r';
	const char * hehe = &faction;

	FILE* fp = fopen(fname, hehe);

	char readBuffer[65536];
	FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	Document para;
	para.ParseStream(is);

    /* need to pass all the parameters to the device as well */

	checkCudaErrors(cudaMalloc(&d_m, sizeof(size_t)));
	h_m = para["m"].GetInt();
	checkCudaErrors(cudaMemcpy(d_m, &h_m, sizeof(size_t), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_k, sizeof(size_t)));
	h_k = para["k"].GetInt();
	checkCudaErrors(cudaMemcpy(d_k, &h_k, sizeof(size_t), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_T, sizeof(size_t)));
	h_T = para["T"].GetInt();
	checkCudaErrors(cudaMemcpy(d_T, &h_T, sizeof(size_t), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_h, sizeof(float)));
	h_h = para["h"].GetDouble();
	checkCudaErrors(cudaMemcpy(d_h, &h_h, sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_r, sizeof(float)));
	h_r = para["r"].GetDouble();
	checkCudaErrors(cudaMemcpy(d_r, &h_r, sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_c, sizeof(float)));
	h_c = para["c"].GetDouble();
	checkCudaErrors(cudaMemcpy(d_c, &h_c, sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_theta, sizeof(float)));
	h_theta = para["theta"].GetDouble();
	checkCudaErrors(cudaMemcpy(d_theta, &h_theta, sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_s, sizeof(float)));
	h_s = para["s"].GetDouble();
	checkCudaErrors(cudaMemcpy(d_s, &h_s, sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_alpha, sizeof(float)));
	h_alpha = para["alpha"].GetDouble();
	checkCudaErrors(cudaMemcpy(d_alpha, &h_alpha, sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_maxhold, sizeof(size_t)));
	h_maxhold = para["maxhold"].GetInt();
	checkCudaErrors(cudaMemcpy(d_maxhold, &h_maxhold, sizeof(size_t), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_lambda, sizeof(float)));
	h_lambda = para["lambda"].GetDouble();
	checkCudaErrors(cudaMemcpy(d_lambda, &h_lambda, sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_min_demand, sizeof(size_t)));
	h_min_demand = para["min_demand"].GetInt();
	checkCudaErrors(cudaMemcpy(d_min_demand, &h_min_demand, sizeof(size_t), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_max_demand, sizeof(size_t)));
	h_max_demand = para["max_demand"].GetInt();
	checkCudaErrors(cudaMemcpy(d_max_demand, &h_max_demand, sizeof(size_t), cudaMemcpyHostToDevice));


	h_demand_distribution = (float *)malloc((h_max_demand - h_min_demand) * sizeof(float));

	const Value& demand_array = para["demand_distribution"];
	for (int i = 0; i < h_max_demand - h_min_demand; ++i){
		h_demand_distribution[i] = demand_array[i].GetDouble();
	}


	fclose(fp);

	return 0;
}
