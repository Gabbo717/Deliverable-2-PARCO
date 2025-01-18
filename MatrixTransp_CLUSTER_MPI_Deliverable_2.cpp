#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

void generateMatrix(std::vector<std::vector<double>> &matrix, int n) {
    srand(time(0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
}

bool checkSymMPI(const std::vector<std::vector<double>> &matrix, int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rowsPerProcess = n / size;
    int extraRows = n % size;

    
    int localRows = rowsPerProcess + (rank < extraRows ? 1 : 0);
    int startRow = rank * rowsPerProcess + std::min(rank, extraRows);

    
    std::vector<double> localMatrix(localRows * n);

    
    if (rank == 0) {
        std::vector<double> flatMatrix(n * n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                flatMatrix[i * n + j] = matrix[i][j];
            }
        }

        std::vector<int> sendCounts(size), displacements(size);
        for (int i = 0; i < size; ++i) {
            sendCounts[i] = (rowsPerProcess + (i < extraRows ? 1 : 0)) * n;
            displacements[i] = i * rowsPerProcess * n + std::min(i, extraRows) * n;
        }

        MPI_Scatterv(flatMatrix.data(), sendCounts.data(), displacements.data(),
                     MPI_DOUBLE, localMatrix.data(), localRows * n,
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_DOUBLE, localMatrix.data(), localRows * n,
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    
    bool localSymmetry = true;
    for (int i = 0; i < localRows; ++i) {
        int globalRow = startRow + i;
        for (int j = 0; j < n; ++j) {
            if (std::abs(localMatrix[i * n + j] - matrix[j][globalRow]) > 1e-9) {
                localSymmetry = false;
                break;
            }
        }
        if (!localSymmetry) break;
    }

    
    bool globalSymmetry = false;
    MPI_Allreduce(&localSymmetry, &globalSymmetry, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    return globalSymmetry;
}

void transposeMPI(const std::vector<std::vector<double>> &matrix,
                  std::vector<std::vector<double>> &result, int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int baseRows = n / size; 
    int leftover = n % size; 

    int myRows = baseRows + (rank < leftover ? 1 : 0);

    std::vector<int> myRowsPerProc(size);
    for (int i = 0; i < size; ++i) {
        myRowsPerProc[i] = baseRows + (i < leftover ? 1 : 0);
    }

    std::vector<int> counts(size), displs(size);
    {
        int currentDisp = 0;
        for (int i = 0; i < size; ++i) {
            counts[i] = myRowsPerProc[i] * n;  
            displs[i] = currentDisp;
            currentDisp += counts[i];
        }
    }

    std::vector<double> globalBuffer(n * n);

    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                globalBuffer[i * n + j] = matrix[i][j];
            }
        }
    }

    std::vector<double> localMatrix(myRows * n);

    MPI_Scatterv(globalBuffer.data(), counts.data(), displs.data(),
                 MPI_DOUBLE, localMatrix.data(), myRows * n,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> transposedLocalMatrix(myRows * n);
    for (int i = 0; i < myRows; ++i) {
        for (int j = 0; j < n; ++j) {
            transposedLocalMatrix[j * myRows + i] = localMatrix[i * n + j];
        }
    }

    std::vector<double> globalResult(n * n);

    {
        int currentDisp = 0;
        for (int i = 0; i < size; ++i) {
            counts[i] = n * myRowsPerProc[i];
            displs[i] = currentDisp;
            currentDisp += counts[i];
        }
    }

    MPI_Gatherv(transposedLocalMatrix.data(), n * myRows, MPI_DOUBLE,
                globalResult.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<int> rowOffsetPerProc(size);
        rowOffsetPerProc[0] = 0;
        for (int i = 1; i < size; ++i) {
            rowOffsetPerProc[i] = rowOffsetPerProc[i - 1] + myRowsPerProc[i - 1];
        }

        for (int p = 0; p < size; ++p) {
            int offsetInGlobal = displs[p];
            int startCol       = rowOffsetPerProc[p];
            for (int i = 0; i < myRowsPerProc[p]; ++i) {
                for (int j = 0; j < n; ++j) {
                    double val = globalResult[offsetInGlobal + j * myRowsPerProc[p] + i];
                    result[j][startCol + i] = val;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 8; // default
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }

    int num_trials = 1; // default
    if (argc > 2) {
        num_trials = std::atoi(argv[2]);
    }

    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> transposedMatrix(n, std::vector<double>(n, 0.0));

    for (int iter = 0; iter < num_trials; ++iter) {
        if (rank == 0) {
            generateMatrix(matrix, n);
        }

        double startTime = MPI_Wtime();
        checkSymMPI(matrix, n);
        double endTime = MPI_Wtime();

        if (rank == 0) {
            std::cout << (endTime - startTime) << std::endl;
        }

        startTime = MPI_Wtime();
        transposeMPI(matrix, transposedMatrix, n);
        endTime = MPI_Wtime();

        if (rank == 0) {
            std::cout << (endTime - startTime) << std::endl;
        }

    }

    MPI_Finalize();
    return 0;
}
