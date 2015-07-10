#include <cmath>
#include <fstream>
#include <iostream>
#include "timer.h"
#include "parameters.h"
#include "utils.h"
#include "support.h"
#include "model.h"
#include "loadParams.h"
using namespace std;

size_t  valueTablesLength;


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

inline bool is_file_exist (const std::string& name) {
    ifstream f(name.c_str());
    if (f.good()) {
        f.close();
        return true;
    } else {
        f.close();
        return false;
    }
}

const char * execCMD(string cmd){
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "ERROR";
    char buffer[128];
    std::string result = "";
    while(!feof(pipe)) {
        if(fgets(buffer, 128, pipe) != NULL)
            result += buffer;
    }
    pclose(pipe);
    result.resize(result.size() - 1);
    return result.c_str();
}


int main(int argc, char ** argv){
    /* load the global variables */
    string filename = "../param.json";
    if ( argc < 2){
        cerr << "\e[38;5;57mWarning: no input." << "\e[m" << endl << "Looking for the default config file : " << filename << endl;
        if ( !is_file_exist(filename)){
            cerr << "\e[38;5;196mCannot find " << execCMD("pwd") << "/" << filename << ", abort.\e[m";
            return -1;
        }

    }
    else if ( !is_file_exist(string(argv[1])) ){
        cerr << "\e[38;5;196mError: cannot find file.\e[m" << endl << "Cannot find " <<  execCMD("pwd") << "/" <<  argv[1] << ", please check the filename and the path, abort." << endl;
        return -1;
    }
    else {
        filename = argv[1];
    }

    cerr << "Config file found." << endl;
    loadParams(filename.c_str());
    checkParams();

	/* declare variables */
    /* system features */

    cudaInfoStruct cudainfo;
    cudainfo.deviceCount = 0;
    cudainfo.numBlocks = 0;
    cudainfo.numThreadsPerBlock = 0;
    gatherSystemInfo(&cudainfo);

    cout << "System Configuration : " << endl
         << "   Number of CUDA Devices : " << "\e[38;5;166m" << cudainfo.deviceCount << "\e[m" << endl
         << "   Number of cores : " << "\e[38;5;166m" << cudainfo.numBlocks << "\e[m" << endl
         << "   Number of threads per core : " << "\e[38;5;166m" <<  cudainfo.numThreadsPerBlock << "\e[m" << endl;

	// host tables
	float * h_valueTable;

	// device tables
	// two tables in total for cross updating and another table to store the random distribution
	float ** d_valueTables;
    d_valueTables = (float **)malloc(2 * sizeof(float *));
    float * d_randomTable;


    valueTablesLength = pow(h_k, h_m);
    cout <<"valueTablesLength: " <<  valueTablesLength << endl;



	/* memory allocation */
	h_valueTable = (float*)malloc( valueTablesLength * sizeof(float) );

    cerr << "1" << endl;


    deviceTableInit(2, 					// total number of tables
    				d_valueTables,
    				valueTablesLength,
                    &cudainfo);
    cerr << "2" << endl;

    deviceTableInit(1,
                    &d_randomTable,
                    (h_max_demand - h_min_demand),
                    &cudainfo);
    cerr << "3" << endl;

    passToDevice(h_demand_distribution, d_randomTable, (h_max_demand - h_min_demand));
    cerr << "4" << endl;


    /* evalueate the value of the table with the given parameters and the given policy f */
    // the policy is : the depletion amount is always zero except the first day
    size_t currentTableIdx = 0;      // the index of table to be updated in next action

    // /*first init make one of the d_valueTables to the edge values*/
    presetValueTable(d_valueTables[currentTableIdx], valueTablesLength, &cudainfo);
    currentTableIdx = 1 - currentTableIdx;
    cerr << "5" << endl;

    readFromDevice(h_valueTable, d_valueTables[1 - currentTableIdx], valueTablesLength);
    cerr << "6" << endl;



    // /* T periods in total */
    for ( size_t iPeriod = h_T; iPeriod > 0; --iPeriod){
      if(iPeriod != 1){
        valueTableUpdateWithPolicy( d_valueTables, currentTableIdx, 0, d_randomTable, &cudainfo);
        currentTableIdx = 1 - currentTableIdx;
      }
      else{
        // first calculate the expect demand for each day
        float expectDemand = 0;
        for (int i = h_min_demand; i < h_max_demand - h_min_demand + 1; ++i){
            expectDemand += i * h_demand_distribution[i];
        }
                // cerr << " the expected demand : " << endl << (size_t)ceil(expectDemand) << endl;
        valueTableUpdateWithPolicy( d_valueTables, currentTableIdx, (size_t)ceil(expectDemand), d_randomTable, &cudainfo);
      }
    }
    readFromDevice(h_valueTable, d_valueTables[1 - currentTableIdx], valueTablesLength);
    printResult(h_valueTable, valueTablesLength, "Value table");

    return 0;
}

