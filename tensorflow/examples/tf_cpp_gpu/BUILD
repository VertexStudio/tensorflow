# Description:
#   TensorFlow C++ wrapper for Venice.

package(
    default_visibility = ["//tensorflow:internal"],
)

load("//tensorflow:tensorflow.bzl", "tf_native_cc_binary")

tf_native_cc_binary(
    name = "tf_cpp_gpu.dll",
    linkshared = 1,
    srcs = [
        "model.cc",
        "model.h",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)
