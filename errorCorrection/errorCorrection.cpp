// errorCorrection.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <vector>
#include <cmath>
#include <iostream>
#include <Eigen/dense>
#include <chrono>
#include <thread>
using namespace std;
using namespace Eigen;

template <typename T> 
void operator <<(ostream& os, vector<T> arr) {
    for (auto temp : arr)
        os << ", " << temp;
    os << endl;
    //return os;
}

struct learningData {
    VectorXd weights; 
    vector<double> mseArr; 
};

double randDouble() { return (double)rand() / (double)RAND_MAX; }

VectorXd hebbFunction(MatrixXd inputs, MatrixXd outputs) {
    // function: calculates the weights of a sample set 
    // produces a matrix of weights. Its dimensions are (nxm) where n = inputs.cols() and m = outputs.rows()   
    vector<MatrixXd> arrIn = {};
    vector<MatrixXd> arrOut = {};
    double N = outputs.rows(); // sample size for this case

    // populating vector of correspondent input and output samples 
    // outputs
    for (int x = 0; x < outputs.rows(); x++) {
        arrOut.push_back(MatrixXd(1, outputs.cols()));
        for (int y = 0; y < outputs.cols(); y++) {
            arrOut[x](0, y) = outputs(x, y);
        }
    }

    // inputs
    for (int x = 0; x < inputs.rows(); x++) {
        arrIn.push_back(MatrixXd(1, inputs.cols()));
        for (int y = 0; y < inputs.cols(); y++) {
            arrIn[x](0, y) = inputs(x, y);
        }
    }

    vector<MatrixXd> products = {};
    for (int i = 0; i < inputs.rows(); i++) {
        // append x[k]T * y[k] to a vector of products
        products.push_back(arrIn[i].transpose() * arrOut[i]);
        // since the inputs are in row major form, the inputs matrix must be transposed
    }

    MatrixXd Sum = MatrixXd::Zero(products[0].rows(), products[0].cols());

    // loop to sum each dotProduct
    for (auto& temp : products) {
        Sum += temp;
    }

    Sum /= N;

    VectorXd weights(Sum.rows()+1);
    weights(0) = 1;
    for (int i = 0; i < Sum.rows(); i++) 
        weights(i+1) = Sum(i,0);
    return weights;
}

// z : weightedSum 
// tanh = formats weightedSum to a continuous value between -1 and 1 
// double tanh(const double z) { return (exp(z) - exp(-z)) / (exp(z) + exp(-z)); }
//  tanh() provided by <cmath> addresses divide-by-zero errors

void updateWeights(VectorXd inputs, VectorXd& weights, const double learningRate,
    const double delta, const double marginOfErr) {
    weights(0) += learningRate * delta;
    for (int i = 1; i < inputs.size(); i++) {
        weights(i) += learningRate * delta * inputs(i - 1);
    }
    //cout << "UPDATED WEIGHTS: \n" << weights << endl;
}

// pairs a tanh(z) to their respective class 
double classify(double neuronOut, int classes) {
    double interval = (double)2 / classes;
    //cout << "interval: " << interval << endl;
    int bin = 0;
    //cout << "NeuronOut: " << neuronOut << endl;

    for (double i = -1.0; i < 1.0; i += interval) {
      //  cout << "interval: \n Min: " << i << "  Max: " << i + interval << endl;
        if (i < 1 - interval) {
            if (i <= neuronOut && neuronOut < i + interval)
                return bin;
        }
        else if (i <= neuronOut && neuronOut <= i + interval)
            return bin;
        bin++;
    }
    return bin;
}
double classify(double neuronOut, int classes, bool output) {
    //bool output is irrelevant
    double interval = (double)2 / classes;
    cout << "\ninterval: " << interval << endl;
    int bin = 0;
    cout << "NeuronOut: " << neuronOut << endl;

    for (double i = -1.0; i < 1.0; i += interval) {
        cout << "interval: \n Min: " << i << "  Max: " << i + interval << endl;
        if (i < 1 - interval) {
            if (i <= neuronOut && neuronOut < i + interval)
                return bin;
        }
        else if (i <= neuronOut && neuronOut <= i + interval)
            return bin;
        bin++;
    }
    return bin;
}

void testWeights(VectorXd weights, MatrixXd inputs, VectorXd dOuts, int classes) {
    VectorXd output(inputs.rows());

    for (int x = 0; x < dOuts.size(); x++) {
        VectorXd sample = inputs.row(x);
        double sum = weights(0);
        for (int i = 1; i < inputs.cols() + 1; i++)
            sum += weights(i) * sample(i - 1);
        output(x) = classify(tanh(sum), classes, true);
        cout << "CLASS: " << output(x)<<endl;
    }

    cout << "OUTPUT FROM EXTRACTED WEIGHTS: \n" << output << endl;

    if (dOuts.isApprox(output))
        cout << "\033[32mExpected output found\033[0m" << endl;
    else
        cout << "\033[31mExpected output not found\033[0m" << endl;
}

// core function that extracts weights from a learning set 
learningData learnWeights(MatrixXd learningSet, int classes, bool goRMSE, bool randWeights) {
    MatrixXd inputs = learningSet.block(0, 0, learningSet.rows(), learningSet.cols() - 1);
    cout << "INPUTS: \n" << inputs << endl;

    VectorXd dOuts = learningSet.col(learningSet.cols() - 1);
    cout << "DESIRED OUTS: \n" << dOuts << endl;
    cout << "\n+++++\n";

    VectorXd weights(learningSet.cols());

    //populating initial weights 
    if (randWeights) {
        weights(0) = 1; // starting with bias 
        for (int y = 1; y < weights.size(); y++) {
            weights(y) = randDouble();
        }
        cout << "Randomly Generated Weights: \n" << weights << endl;
        cout << "\n+++++\n";
    }
    else {
        weights = hebbFunction(inputs, (MatrixXd)dOuts);
        cout << "HEBBIAN WEIGHTS: \n" << weights << endl;
        cout << "\n+++++\n";
    }

    //intializing constants
    double learningRate = 1.0 / weights.size();
    learningRate = 0.001;
    const double marginOfErr = 1.0 / classes;

    VectorXd deltaArr(inputs.rows());
    deltaArr.setZero();

    vector<double> errData;
    VectorXd neuronOuts(inputs.rows());

    int iteration = 0;
    const int iterMax = 1000000;
    bool continueLearning = false;

    int vectInd = 0;
    do {
        //printf("EPOCH: %d \n", iteration);
        //cout << "WEIGHTS: \n" << weights << endl;

        for (int iter = 0; iter < inputs.rows(); iter++) {
            VectorXd input = inputs.row(iter);
           // cout << "SAMPLE " << iter << ": \n" << input << endl;
            double delta = deltaArr(iter);

            if (continueLearning)
                updateWeights(inputs.row(iter), weights, learningRate, delta, marginOfErr);

            double sum = weights(0);
            for (int i = 1; i < inputs.cols() + 1; i++) {
                sum += weights(i) * input(i - 1);
            }

            //cout << "WEIGHTED SUM " << iter << ": " << sum << endl;
            neuronOuts[iter] = classify(tanh(sum), classes);

            if (isnan(tanh(sum))) {
                cerr << "INVALID VALUE FROM TANH()" << endl;
                exit(1);
            }

            //printf("tanh(z): %lf  assignedClass: %lf ", tanh(sum), neuronOuts[iter]);
            //if (dOuts(iter) == neuronOuts[iter])
                //printf("\033[32m expected output found \033[0m \n");
            //else
                //printf("\033[31m expected output not found \033[0m \n");

            
            deltaArr[iter] = dOuts[iter] - neuronOuts[iter];
        }
        //cout << "\n+++++\n";
        double MSE = 0;
        for (auto d : deltaArr)
            MSE += pow(d, 2);

        MSE /= inputs.rows();
        double RMSE = sqrt(MSE);
        //neighboring duplicates are not included 
        if (vectInd == 0 || errData[vectInd - 1] != RMSE) {
            //if(vectInd - 1 >= 0)
              //  cout << "errData[vectInd-1]: "<< errData[vectInd - 1] << " Pushed RMSE: " << RMSE << endl;
            errData.push_back(RMSE);
            vectInd++;
        }

        if (RMSE < marginOfErr && goRMSE)
            continueLearning = false;
        else if (MSE < marginOfErr && !goRMSE)
            continueLearning = false;
        else
            continueLearning = true;
        iteration++;
    } while (continueLearning && iteration < iterMax);

    learningData datum;
    datum.weights = weights;
    datum.mseArr = errData;

    cout << "ITERATIONS: " << iteration << endl;
    cout << "\nEXTRACTED WEIGHTS: \n" << weights << endl;
    cout << "\n+++++\n";
    return datum;
}

learningData learnWeights(MatrixXd learningSet, int classes, bool goRMSE, bool randWeights) {
    MatrixXd inputs = learningSet.block(0, 0, learningSet.rows(), learningSet.cols() - 1);
    cout << "INPUTS: \n" << inputs << endl;

    VectorXd dOuts = learningSet.col(learningSet.cols() - 1);
    cout << "DESIRED OUTS: \n" << dOuts << endl;
    cout << "\n+++++\n";

    VectorXd weights(learningSet.cols());

    //populating initial weights 
    if (randWeights) {
        weights(0) = 1; // starting with bias 
        for (int y = 1; y < weights.size(); y++) {
            weights(y) = randDouble();
        }
        cout << "Randomly Generated Weights: \n" << weights << endl;
        cout << "\n+++++\n";
    }
    else {
        weights = hebbFunction(inputs, (MatrixXd)dOuts);
        cout << "HEBBIAN WEIGHTS: \n" << weights << endl;
        cout << "\n+++++\n";
    }

    //intializing constants
    double learningRate = 1.0 / weights.size();
    learningRate = 0.001;
    const double marginOfErr = 1.0 / classes;

    VectorXd deltaArr(inputs.rows());
    deltaArr.setZero();

    vector<double> errData;
    VectorXd neuronOuts(inputs.rows());

    int iteration = 0;
    const int iterMax = 1000000;
    bool continueLearning = false;

    int vectInd = 0;
    do {
        for (int iter = 0; iter < inputs.rows(); iter++) {
            VectorXd input = inputs.row(iter);
            double delta = deltaArr(iter);

            if (continueLearning)
                updateWeights(inputs.row(iter), weights, learningRate, delta, marginOfErr);

            double sum = weights(0);
            for (int i = 1; i < inputs.cols() + 1; i++) {
                sum += weights(i) * input(i - 1);
            }
            neuronOuts[iter] = classify(tanh(sum), classes);
    
            deltaArr[iter] = dOuts[iter] - neuronOuts[iter];
        }
        double MSE = 0;
        for (auto d : deltaArr)
            MSE += pow(d, 2);

        MSE /= inputs.rows();
        double RMSE = sqrt(MSE);

        if (vectInd == 0 || errData[vectInd - 1] != RMSE) {
            errData.push_back(RMSE);
            vectInd++;
        }

        if (RMSE < marginOfErr && goRMSE)
            continueLearning = false;
        else if (MSE < marginOfErr && !goRMSE)
            continueLearning = false;
        else
            continueLearning = true;
        iteration++;
    } while (continueLearning && iteration < iterMax);

    learningData datum;
    datum.weights = weights;
    datum.mseArr = errData;

    cout << "ITERATIONS: " << iteration << endl;
    cout << "\nEXTRACTED WEIGHTS: \n" << weights << endl;
    cout << "\n+++++\n";
    return datum;
}

// REFORMATTED LEARNWEIGHTS() 

learningData learnWeights(MatrixXd inputs, VectorXd dOuts, VectorXd weights, double threshold, int classes) {
    cout << "INPUTS: \n" << inputs << endl;

    cout << "DESIRED OUTS: \n" << dOuts << endl;
    cout << "\n+++++\n";

    //intializing constants
    double learningRate = 1.0 / weights.size();
    learningRate = 0.001;

    VectorXd deltaArr(inputs.rows());
    deltaArr.setZero();

    vector<double> errData;
    VectorXd neuronOuts(inputs.rows());

    int iteration = 0;
    const int iterMax = 1000000;
    bool continueLearning = false;

    int vectInd = 0;
    do {
        //printf("EPOCH: %d \n", iteration);
        //cout << "WEIGHTS: \n" << weights << endl;

        for (int iter = 0; iter < inputs.rows(); iter++) {
            VectorXd input = inputs.row(iter);
            // cout << "SAMPLE " << iter << ": \n" << input << endl;
            double delta = deltaArr(iter);

            if (continueLearning)
                updateWeights(inputs.row(iter), weights, learningRate, delta, threshold);

            double sum = weights(0);
            for (int i = 1; i < inputs.cols() + 1; i++) {
                sum += weights(i) * input(i - 1);
            }

            //cout << "WEIGHTED SUM " << iter << ": " << sum << endl;
            neuronOuts[iter] = classify(tanh(sum), classes);

            if (isnan(tanh(sum))) {
                cerr << "INVALID VALUE FROM TANH()" << endl;
                exit(1);
            }

            //printf("tanh(z): %lf  assignedClass: %lf ", tanh(sum), neuronOuts[iter]);
            //if (dOuts(iter) == neuronOuts[iter])
                //printf("\033[32m expected output found \033[0m \n");
            //else
                //printf("\033[31m expected output not found \033[0m \n");


            deltaArr[iter] = dOuts[iter] - neuronOuts[iter];
        }
        //cout << "\n+++++\n";
        double MSE = 0;
        for (auto d : deltaArr)
            MSE += pow(d, 2);

        MSE /= inputs.rows();
        double RMSE = sqrt(MSE);
        //duplicate values are not included 
        if (vectInd == 0 || errData[vectInd - 1] != RMSE) {
            //if(vectInd - 1 >= 0)
              //  cout << "errData[vectInd-1]: "<< errData[vectInd - 1] << " Pushed RMSE: " << RMSE << endl;
            errData.push_back(RMSE);
            vectInd++;
        }

        if (RMSE < threshold)
            continueLearning = false;
        else
            continueLearning = true;
        iteration++;
    } while (continueLearning && iteration < iterMax);

    learningData datum;
    datum.weights = weights;
    datum.mseArr = errData;

    cout << "ITERATIONS: " << iteration << endl;
    cout << "\nEXTRACTED WEIGHTS: \n" << weights << endl;
    cout << "\n+++++\n";
    return datum;
}


int main() {
    int classes = 3;
    MatrixXd samples(3, 4);
    samples << 0, 2, 1, 0,
        1, 2, 3, 1,
        2, 1, 3, 2;
    
    double threshold = 1.0 / classes; 

    MatrixXd inputs = samples.block(0, 0, samples.rows(), samples.cols() - 1);
    VectorXd outs = samples.col(samples.cols() - 1);
    double learningRate = 1.0 / samples.cols();
    
    VectorXd randWeights(samples.cols());
    randWeights.setRandom(); randWeights(0) = 1;//bias

    VectorXd hebbWeights = hebbFunction(inputs, (MatrixXd)outs);


   //learningData datum = learnWeights(samples, classes, true, false);
   //testWeights(datum.weights, inputs, outs, classes);
   //cout<< "EXTRACTED RMSE VECTOR: " << datum.mseArr;

    learningData datum = learnWeights(inputs, outs, randWeights, threshold, 3);

}
