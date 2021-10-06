#include <iostream>
#include <vector>
#include "omp.h"
#include <fstream>
#include <sstream>
#include <math.h>
#include <limits>

using namespace std;

size_t numThreads = 16;


//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//     Сначала идут вспомогательные функции, не         //
//     относящиеся к самому алгоритму (вывод и ввод)    //
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////



void printMatrixToFile(const string &path, vector<vector<double>> const& m) {
    // Выводим матрицу в файл
    // Вход:
    //     path - строка,путь к файлу
    //     m - вещественная матрица, которую выводим
    // Выход:
    ofstream file(path);
    for(int i = 0; i < m.size(); ++i) {
        for(int j = 0; j < m[0].size(); ++j) {
            file << m[i][j];
            if  (j < (m[0].size() - 1)) {
                file << ',';
            }
        }
        file << endl;
    }
}

void printVectorToFile(const string &path, vector<double> const& v) {
    // Выводим вектор в файл
    // Вход:
    //     path - строка,путь к файлу
    //     v - вещественный вектор, который выводим
    // Выход:
    ofstream file(path);
    for(int i = 0; i < v.size(); ++i) {
        file << v[i];
        if  (i < (v.size() - 1)) {
            file << ',';
        }
    }
    file << endl;
}

void printTimeToFile(const string &path, double d) {
    // Выводим время в файл
    // Вход:
    //     path - строка,путь к файлу
    //     d - вещественное число, которое выводим
    // Выход:
    ofstream file(path);
    file << d << endl;
}

vector<vector<double>> readMatrixFromFile(const string &path) {
    // Считываем матрицу из файла
    // Вход:
    //     path - строка,путь к файлу
    // Выход:
    //     вещественная матрица
    vector<vector<double>> matrix;
    string row;
    string token;
    ifstream file;
    file.open(path, std::ios::out);
    while (!file.eof()) {
        getline(file, row);
        if (file.bad() || file.fail()) {
            break;
        }
        vector<double> values;
        stringstream ss(row);
        while (getline(ss, token, ',')) {
            values.push_back(atof(token.c_str()));
        }
        matrix.push_back(values);
    }
    return matrix;
}

vector<double> genRandVec(size_t n) {
    // Генерируем случайный вектор
    // Вход:
    //     n - длина вектора
    // Выход:
    //     случайный вектор
    vector<double> res(n, 0);
    for (size_t i = 0; i < n; i++) {
        res[i] = ((double)rand() / RAND_MAX) * 2000 - 1000;
    }
    return res;
}

void genRandMatrixToFile(size_t m, size_t n, const string &path="test.csv") {
    // Генерируем случайную матрицу в файл
    // Вход:
    //     m - количество строк в матрице
    //     n - количество столбцов
    //     path - строка,путь к файлу
    // Выход:
    ofstream outfile(path);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            outfile << ((double)rand() / RAND_MAX) * 2000 - 1000;
            if (j < n - 1) {
                cout << ',';
            }
        }
        outfile << std::endl;
    }
}

vector<vector<double>> genRandMatrix(size_t m, size_t n) {
    // Генерируем случайную матрицу
    // Вход:
    //     m - количество строк в матрице
    //     n - количество столбцов
    // Выход:
    //     случайная матрица
    vector<vector<double>> matrix(m, vector<double>(n));
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            matrix[i][j] = ((double)rand() / RAND_MAX) * 2000 - 1000;
        }
    }
    return matrix;
}

void printVector(vector<double> const&v) {
    // Вывод на экран вектора
    // Вход:
    //     v - вектор
    // Выход:
    for(int i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

void printMatrix(vector<vector<double>> const& m) {
    // Вывод на экран матрицы
    // Вход:
    //     m - матрица
    // Выход:
    for(int i = 0; i < m.size(); ++i) {
        for(int j = 0; j < m[0].size(); ++j) {
            cout << m[i][j] << ' ';
        }
        cout << endl;
    }
}


//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//          Блок с основными функциями                  //
//                 алгоритма                            //
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////


vector<vector<double>> transposeMatrix(vector<vector<double>> const& m) {
    // Транспонирование матрицы
    // Вход:
    //     m - матрица
    // Выход:
    //     транспонированная матрица
    vector<vector<double>> mT(m[0].size(), vector<double>(m.size()));
    #pragma omp parallel
    {
        #pragma omp for
        for(int i = 0; i < m.size(); ++i) {
            for(int j = 0; j < m[0].size(); ++j) {
                mT[j][i] = m[i][j];
            }
        }
    }
    return mT;
}

double scalarMultiply(vector<double> const& v1, vector<double> const& v2) {
    // Скалярное произведение векторов
    // Вход:
    //     v1 - первый вектор
    //     v2 - второй вектор
    // Выход:
    //     скалярное произведение векторов v1 и v2
    if (v1.size() != v2.size()) {
      cerr << "ERROR: incorrectly sized vectors in scalarMultiply\n";
      exit(EXIT_FAILURE);
    }

    double res = 0.;
    for(int i = 0; i < v1.size(); ++i) {
        res += v1[i] * v2[i];
    }
    return res;
}

vector<double> matrixVectorMultiply(vector<vector<double>> const& m, vector<double> const& v) {
    // Произведение матрицы на вектор
    // Вход:
    //     m - матрица
    //     v - вектор
    // Выход:
    //     результирующий вектор
    if (m[0].size() != v.size()) {
      cerr << "ERROR: incorrectly sized matrix or vector in matrixVectorMultiply\n";
      exit(EXIT_FAILURE);
    }

    vector<double> res(m.size(), 0);
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for (int i = 0; i < m.size(); ++i) {
            for (int j = 0; j < m[0].size(); ++j) {
                res[i] += m[i][j] * v[j];
            }
        }
    }
    return res;
}

vector<vector<double>> matrixMultiply(vector<vector<double>> const& m) {
    // Матричное умножение m^T * m
    // Вход:
    //     m - матрица
    // Выход:
    //     результирующая матрица
    vector<vector<double>> res(m[0].size(), vector<double>(m[0].size()));
    int i, j, k;
    double prod;
    #pragma omp parallel for shared(m, res) private(i, j, k, prod) num_threads(numThreads)
    for (i = 0; i < m[0].size(); i++) {
        for (j = i; j < m[0].size(); j++) {
            for (k = 0; k < m.size(); k++) {
                prod = (m[k][i] * m[k][j]);
                res[i][j] += prod;
                if (i != j) {
                    res[j][i] += prod;
                }
            }
        }
    }
    return res;
}

vector<vector<double>> vecToMatrix(vector<double> const& v) {
    // Добавление оси вектору
    // Вход:
    //     v - вектор
    // Выход:
    //     результирующая матрица
    vector<vector<double>> res(1, vector<double>(v.size()));
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for (int i = 0; i < v.size(); i++) {
            res[0][i] = v[i];
        }
    }
    return res;
}

vector<double> vecDivToNum(vector<double> const& v, double num) {
    // Деление вектора на число
    // Вход:
    //     v - вектор
    // Выход:
    //     результирующий вектор
    vector<double> res(v.size());
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for (int i = 0; i < v.size(); i++) {
            res[i] = v[i] / num;
        }
    }
    return res;
}

vector<vector<double>> matrMulToNum(vector<vector<double>> const& m, double num) {
    // Умножение матрицы на число
    // Вход:
    //     m - матрица
    //     num - вещественное число
    // Выход:
    //     результирующая матрица
    vector<vector<double>> res(m.size(), vector<double>(m[0].size()));
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for(int i = 0; i < m.size(); ++i) {
            for(int j = 0; j < m[0].size(); ++j) {
                res[i][j] = m[i][j] * num;
            }
        }
    }
    return res;
}

vector<vector<double>> difMatrix(vector<vector<double>> const& m1, vector<vector<double>> const& m2) {
    // Разница матриц
    // Вход:
    //     m1 - матрица 1
    //     m2 - матрица 2
    // Выход:
    //     результирующая матрица
    if ((m1.size() != m2.size()) || (m1[0].size() != m2[0].size())) {
      cerr << "ERROR: incorrectly sized matrix or vector in difMatrix\n";
      exit(EXIT_FAILURE);
    }
    vector<vector<double>> res(m1.size(), vector<double>(m1[0].size()));
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for(int i = 0; i < m1.size(); ++i) {
            for(int j = 0; j < m1[0].size(); ++j) {
                res[i][j] = m1[i][j] -  m2[i][j];
            }
        }
    }
    return res;
}

void findSZandSV(vector<vector<double>> const& matrix, vector<vector<double>> &evSV, vector<double> &evSZ, int k, int maxIter = 500, double accuracy = 0.000000001) {
    // Нахождение собственных векторов и собственных значений матрицы matrix^T * matrix
    // Вход:
    //     matrix - матрица объектов
    //     evSV - матрица, которую заполним собственными векторами
    //     evSZ - вектор, который заполним собственными значениями
    //     k - количество собственных значений и собственных векторов
    //     maxIter - максимальное количество итераций
    //     accuracy - точность прибижения
    // Выход:
    vector<vector<double>> mTm = matrixMultiply(matrix);
    vector<double> rOld;
    vector<double> rNew;
    double muOld;
    double muNew;
    vector<vector<double>> lrr;
    vector<double> tmp;
    double scalarTmp;
    double scalarROld;
    for (int ev = 0; ev < k; ++ev) {
        rOld = genRandVec(mTm.size());
        muOld = numeric_limits<double>::lowest();
        for (int i = 0; i < maxIter; ++i) {
            tmp = matrixVectorMultiply(mTm, rOld);
            scalarTmp = sqrt(scalarMultiply(tmp,tmp));
            scalarROld = scalarMultiply(rOld,rOld);
            if ((scalarROld == 0) || (scalarTmp == 0)){
                rNew = genRandVec(mTm.size());
                muNew = 0;
                rNew = vecDivToNum(rNew,sqrt(scalarMultiply(rNew,rNew)));
                break;
            }
            rNew = vecDivToNum(tmp,sqrt(scalarMultiply(tmp,tmp)));
            muNew = scalarMultiply(rOld,tmp) / scalarMultiply(rOld,rOld);
            if (fabs(muOld - muNew) <= accuracy) {
                break;
            } else {
                rOld = rNew;
                muOld = muNew;
            }
        }
        evSZ.push_back(muNew);
        evSV.push_back(rNew);
        lrr = matrMulToNum(matrixMultiply(vecToMatrix(rNew)),muNew);
        mTm = difMatrix(mTm, lrr);
    }
}

int main() {
    vector<vector<double>> evSV;
    vector<vector<double>> matrix = readMatrixFromFile("data.csv");
    vector<double> evSZ;
    double s = omp_get_wtime();
    findSZandSV(matrix, evSV, evSZ, 2);
    double t = omp_get_wtime();
    cout << t - s << endl;
    return 0;
}
