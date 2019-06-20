#include <fstream>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "model.h"

typedef Eigen::Matrix<Eigen::half, -1, -1> MatrixXh;

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

DLLEXPORT void RunModelGPU(void* session, int batch, Eigen::MatrixXf& input_left, Eigen::MatrixXf& input_right, float* mask) {
	return Model::RunModel(session, batch, input_left, input_right, mask);
}

DLLEXPORT void* LoadModelGPU(const char* graphpath) {
	return Model::LoadModel(graphpath);
}

DLLEXPORT void FreeModelGPU(void* session) {
	return Model::FreeModel(session);
}


// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(string graph_file_name, std::unique_ptr<tensorflow::Session>* session, tensorflow::SessionOptions* sessionoptions) 
{
  tensorflow::GraphDef graph_def;

  Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) 
  {
    return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
  }
  
  session->reset(tensorflow::NewSession(*sessionoptions));
  
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) 
  {
    return session_create_status;
  }
  return Status::OK();
}



void* Model::LoadModel(string graphpath)
{
	std::unique_ptr<tensorflow::Session> *session = new std::unique_ptr<tensorflow::Session>;
	tensorflow::SessionOptions* sessionoptions = new  tensorflow::SessionOptions();
	tensorflow::GPUOptions gpuoptions;
	sessionoptions = new  tensorflow::SessionOptions();
	//gpuoptions.set_per_process_gpu_memory_fraction(0.1);
	gpuoptions.set_allow_growth(true);
	sessionoptions->config.set_allocated_gpu_options(&gpuoptions);

	//string graph_path = tensorflow::io::JoinPath(graphpath);
	//std::unique_ptr<tensorflow::Session> *session1 = (std::unique_ptr<tensorflow::Session>*)session;
	Status load_graph_status = LoadGraph(graphpath, session, sessionoptions);
	if (!load_graph_status.ok())
	{
		LOG(ERROR) << load_graph_status;
	}

	return session;
}


void Model::RunModel(void* session, int batch, Eigen::MatrixXf& input_left, Eigen::MatrixXf& input_right, float * mask)
{
	string input_layer = "input_1";
	string output_layer = "Sigmoid";

	Tensor keras_learning_phase = Tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
	keras_learning_phase.scalar<bool>()() = 0;

	const Tensor& klp = keras_learning_phase;

	int col = 513, tsteps = 25;
	Tensor resized_tensor = Tensor(tensorflow::DT_HALF, tensorflow::TensorShape({ batch,col,tsteps,1 }));
	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < tsteps; j++)
		{
			resized_tensor.tensor<Eigen::half, 4>()(0, i, j, 0) = (Eigen::half)input_left(i, j);
		}
	}
	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < tsteps; j++)
		{
			resized_tensor.tensor<Eigen::half, 4>()(1, i, j, 0) = (Eigen::half)input_right(i, j);
		}
	}
	
	std::vector<Tensor> outputs;
	std::unique_ptr<tensorflow::Session> *session_casted = (std::unique_ptr<tensorflow::Session>*)session;
	Status run_status = (*session_casted)->Run({ { input_layer, resized_tensor },{ "keras_learning_phase" ,klp } }, { output_layer }, {}, &outputs);

	if (!run_status.ok()) {
		LOG(ERROR) << run_status;
		return;
	}

	for (int i = 0; i < batch; i++)
	{
		for (int j = 0; j < col*4; j++)
		{
			mask[j + i*col*4] = (float)outputs[0].tensor<Eigen::half, 2>()(i, j);
		}
	}
}

void Model::FreeModel(void* session)
{
	std::unique_ptr<tensorflow::Session> *session_casted = (std::unique_ptr<tensorflow::Session>*)session;
	tensorflow::Session* tf_session = session_casted->release();
	tf_session->Close();
	delete tf_session;
	delete session_casted;
}

