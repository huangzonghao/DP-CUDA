#include <cmath>
#include <fstream>
#include <iostream>
#include <cstring>
#include <ctime>
#include "timer.h"
#include "parameters.h"
#include "utils.h"
#include "support.h"
#include "model.h"
#include "loadParams.h"
using namespace std;

size_t  valueTablesLength;

inline bool do_file_exist (char const * name) {
    ifstream f(name);
    if (f.good()) {
        f.close();
        return true;
    } else {
        f.close();
        return false;
    }
}

const char * execCMD(char const * cmd){
    FILE* pipe = popen(cmd, "r");
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


int main(int argc, char ** argv){
    /* load the global variables */
    string inputfile = "../param.json";
    string outputfile = "output.txt";
    string outputformat = "normal";

    if ( argc < 3){
        cout << "\e[38;5;57mWarning: " << "\e[mInsufficient input!" << endl;
        cout << "Usage : [param file] [output file] [output format]" << endl << endl << endl;

        cout << "Using the defaults for the missing parameters" << endl;
    }

    for (int i = 1; i < argc; ++i){
        switch (i){
            case 1:
                inputfile = string(argv[i]);
                break;
            case 2:
                outputfile = string(argv[i]);
                break;
            case 3:
                outputformat = string(argv[i]);
                break;
        }
    }
    cout << "The selected settings : " << endl
        << "    Input file : " << "\e[38;5;51m" << inputfile << endl
        << "    \e[mOutput file : " << "\e[38;5;51m" << outputfile << endl
        << "    \e[mOutput format : " << "\e[38;5;51m" << outputformat << "\e[m" << endl;

    cout << "Looking for the config file : " << inputfile << endl;
    if ( !do_file_exist(inputfile.c_str())){
        cerr << "\e[38;5;196mCannot find " << execCMD("pwd") << "/" << inputfile << ", abort.\e[m" << endl;
        return -1;
    }
    cout << "Config file found." << endl;

    if (outputformat != "csv" && outputformat != "normal"){
        cout << endl << "\e[38;5;57mWarning: \e[m" << "Output file format invalid, using csv." << endl;
        outputformat = "csv";
    }



    loadParams(inputfile.c_str());
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
        << "   Number of threads per core : " << "\e[38;5;166m" <<  cudainfo.numThreadsPerBlock << "\e[m" << endl << endl;

    // host tables
    float * h_valueTable;

    // device tables
    // two tables in total for cross updating and another table to store the random distribution
    float ** d_valueTables;
    d_valueTables = (float **)malloc(2 * sizeof(float *));
    float * d_randomTable;

    valueTablesLength = pow(h_k, h_m);

    /* memory allocation */
    h_valueTable = (float*)malloc( valueTablesLength * sizeof(float) );

    // clock_t clock_begin = clock();

    deviceTableInit(2, 					// total number of tables
            d_valueTables,
            valueTablesLength,
            &cudainfo);

    deviceTableInit(1,
            &d_randomTable,
            (h_max_demand - h_min_demand),
            &cudainfo);

    passToDevice(h_demand_distribution, d_randomTable, (h_max_demand - h_min_demand));


    /* evalueate the value of the table with the given parameters and the given policy f */
    // the policy is : the depletion amount is always zero except the first day
    size_t currentTableIdx = 0;      // the index of table to be updated in next action

    // /*first init make one of the d_valueTables to the edge values*/
    presetValueTable(d_valueTables[currentTableIdx], valueTablesLength, &cudainfo);
    currentTableIdx = 1 - currentTableIdx;
    readFromDevice(h_valueTable, d_valueTables[1 - currentTableIdx], valueTablesLength);



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

    // clock_t clock_end = clock();
    // double elapsed_secs = double(clock_end - clock_begin) / CLOCKS_PER_SEC;

    cout << "Caculation done." << endl;
         // << "Running time \e[38;5;166m" << double(clock_end - clock_begin) << "\e[m s" << endl;
    // cout << endl << "Now start to output.";
    cout << "Now start to output.";
    // printResult(h_valueTable, valueTablesLength, "Value table");
    wirteToFile( h_valueTable, valueTablesLength, outputfile.c_str(), outputformat.c_str() );
    cout << "All processes finished ! " << endl << endl;

    return 0;
}

