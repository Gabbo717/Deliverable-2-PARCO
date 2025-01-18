#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cstring>

using namespace std;

void initializeMatrix(float** M, int n) {
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            M[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

bool checkSym(float** M, int n){
    bool isSymm = true;
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            if(M[i][j] != M[j][i]){
                isSymm = false;
            }
        }
    }
    return isSymm;
}

void matTranspose(float** M, float** T, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            T[i][j] = M[j][i];
        }
    }
}

int main(int argc, char* argv[]){
    if(argc != 2){
        cerr << "Usage: ./Runnable_FileName <matrix_size>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);

    if(n <= 0){
        cerr << "Error: Matrix size must be a positive integer." << endl;
        return 1;
    }

    float** M = new float*[n];
    float** T = new float*[n];
    for(int i = 0; i < n; i++){
        M[i] = new float[n];
        T[i] = new float[n];
    }

    srand(time(0));

    initializeMatrix(M, n);

    auto start = std::chrono::high_resolution_clock::now();
    bool isSymmetric = checkSym(M, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> checkSymTime = end - start;

    start = std::chrono::high_resolution_clock::now();
    matTranspose(M, T, n);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> transposeTime = end - start;

    cout << checkSymTime.count() << endl;
    cout << transposeTime.count() << endl;



    for(int i = 0; i < n; i++){
        delete[] M[i];
        delete[] T[i];
    }
    delete[] M;
    delete[] T;

    return 0;
}


