#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cstring>
#include <omp.h>

using namespace std;

void initializeMatrix(float** M, int n) {
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            M[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

bool checkSymOMP_WorkSharing(float** M, int n){
    bool isSymmetric = true;

    #pragma omp parallel for collapse(2) shared(isSymmetric)
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            if(M[i][j] != M[j][i]){
                
                #pragma omp atomic write
                isSymmetric = false;
                
            }
        }
    }
    return isSymmetric;
}

void matTransposeOMP_WorkSharing(float** M, float** T, int n){
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            T[i][j] = M[j][i];
        }
    }
}

int main(int argc, char* argv[]){
    if(argc != 3){
        cerr << "Usage: ./<runnable_name> <matrix_size> <num_threads>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    if(n <= 0){
        cerr << "Error: Matrix size must be a positive integer." << endl;
        return 1; 
    }

    if(num_threads <= 0){
        cerr << "Error: num_threads must be a positive integer. " << endl;
        return 1;
    }

    omp_set_num_threads(num_threads);

    float** M = new float*[n];
    float** T = new float*[n];
    for(int i = 0; i < n; i++){
        M[i] = new float[n];
        T[i] = new float[n];
    }

    srand(time(0));

    initializeMatrix(M, n);

    auto start = chrono::high_resolution_clock::now();
    bool isSymmetric = checkSymOMP_WorkSharing(M,n);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> checkSymTime_WorkSharing = end - start;

    start = chrono::high_resolution_clock::now();
    matTransposeOMP_WorkSharing(M, T, n);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> transposeTime_WorkSharing = end - start;

    cout << checkSymTime_WorkSharing.count() << endl;
    cout << transposeTime_WorkSharing.count() << endl;

    for (int i = 0; i < n; i++) {
        delete[] M[i];
        delete[] T[i];
    }
    delete[] M;
    delete[] T;

    return 0;
}
