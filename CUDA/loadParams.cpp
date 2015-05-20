#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "parameters.h"
#include <cstdio>

using namespace rapidjson;

int loadParams(string filename){
	FILE* fp = fopen(filename, "r"); 

	char readBuffer[65536];
	FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	Document para;
	para.ParseStream(is);

	m = para["m"].GetInt();
	k = para["k"].GetInt();
	T = para["T"].GetInt();
	h = para["h"].GetDouble();
	r = para["r"].GetDouble();
	c = para["c"].GetDouble();
	theta = para["theta"].GetDouble();
	s = para["s"].GetDouble();
	alpha = para["alpha"].GetDouble();
	maxhold = para["maxhold"].GetInt();
	lambda = para["lambda"].GetDouble();
	min_demand = para["min_demand"].GetInt();
	max_demand = para["max_demand"].GetInt();
	demand_distribution = malloc((max_demand - min_demand) * sizeof(float));
	
	const Value& demand_array = document["demand_distribution"];
	for (int i = 0; i < max_demand - min_demand; ++i){
		demand_distribution[i] = demand_array[i].GetDouble();
	}


	fclose(fp);

	return 0;
}