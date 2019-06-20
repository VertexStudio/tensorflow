#pragma once
#include<iostream>
#include<string>
using namespace std;

class Model
{
	public:
	static void* LoadModel(string graphpath);
	static void RunModel(void* session, int batch, Eigen::MatrixXf& input_left, Eigen::MatrixXf& input_right, float* mask);
	static void FreeModel(void* session);
};

#include <Windows.h>

#define DLLEXPORT __declspec(dllexport)

extern "C" {
	DLLEXPORT void RunModelGPU(void* session, int batch, Eigen::MatrixXf& input_left, Eigen::MatrixXf& input_right, float* mask);
	DLLEXPORT void* LoadModelGPU(const char* graphpath);
	DLLEXPORT void FreeModelGPU(void* session);
}
