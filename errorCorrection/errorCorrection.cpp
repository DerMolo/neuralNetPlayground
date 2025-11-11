// errorCorrection.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <cmath>
#include <iostream>
#include <Eigen/dense>
using namespace std;
using namespace Eigen;

double randDouble() { return (double)rand() / (double)RAND_MAX; }

//z : weightedSum 
//tanh = formats weightedSum to a continuous value between -1 and 1 
double tanh(double z) { return (exp(z) - exp(-z)) / (exp(z) + exp(-z)); }

VectorXd updateWeights(VectorXd inputs, VectorXd weights, double learningRate, double desiredOuts) {

}
//classifies the neuronal output within a bin
//is essentially the activation function 
VectorXd classify(double neuronOut, int classes) {
    double interval = 2 / classes;

}

//core function that extracts weights from a learning set 
VectorXd learnWeights(MatrixXd learningSet, int classes) {

    VectorXd inputs = learningSet.block(0, 0, learningSet.rows(), learningSet.cols() - 1);
    VectorXd desiredOuts = learningSet.col(learningSet.size() - 1);
    VectorXd weights(inputs.cols() + 1);

    weights(0) = 1; //starting with bias 
    for (int y = 1; y < weights.size(); y++) {
        weights(y) = randDouble();
    }
    cout << "Randomly Generated Weights: \n" << weights << endl;
    cout << "\n+++++\n";

    double marginOfErr = 1 / classes;
    double RMSE = 0;

    //weighted sum
    double sum = weights[0] * (delta)
        for (int i = 0; i < weights.size(); i++) {

        }
    //double MSE = pow(RMSE, 2);
}
int main()
{
    std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file


