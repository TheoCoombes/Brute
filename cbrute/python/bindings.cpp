#include <torch/extension.h>
#include "tensor.h"
#include "dtype.h"

PYBIND11_MODULE(_C, m) {
    m.doc() = "brute C++ backend — 1-bit tensors on a PyTorch foundation";

    py::enum_<cbrute::PackDType>(m, "PackDType")
        .value("U8",  cbrute::PackDType::U8)
        .value("U32", cbrute::PackDType::U32)
        .value("U64", cbrute::PackDType::U64);

    py::class_<cbrute::Bit1Tensor>(m, "Bit1Tensor")
        .def(py::init<at::Tensor, std::vector<int64_t>, cbrute::PackDType>(),
             py::arg("data"), py::arg("logical_shape"), py::arg("pack_dtype"))
        .def("data",          &cbrute::Bit1Tensor::data,
             py::return_value_policy::reference_internal)
        .def("logical_shape", &cbrute::Bit1Tensor::logical_shape,
             py::return_value_policy::reference_internal)
        .def("pack_dtype",    &cbrute::Bit1Tensor::pack_dtype)
        .def("pack_width",    &cbrute::Bit1Tensor::pack_width)
        .def("numel",         &cbrute::Bit1Tensor::numel)
        .def("device",        &cbrute::Bit1Tensor::device)
        .def("__repr__", [](const cbrute::Bit1Tensor& t) {
            std::string s = "Bit1Tensor(logical_shape=[";
            for (size_t i = 0; i < t.logical_shape().size(); i++) {
                s += std::to_string(t.logical_shape()[i]);
                if (i + 1 < t.logical_shape().size()) s += ", ";
            }
            s += "])";
            return s;
        });
}
