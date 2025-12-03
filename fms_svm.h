// fms_svm.h: Header file for Support Vector Machine (SVM) functions in the FMS library.
#pragma once
#include <stdexcept>
#include <vector>
#include "svm.h"

namespace fms::svm {

	//enum class svm_type { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };
	//enum class kernel_type { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };

	class problem : public svm_problem {
		int r, c;
		std::vector<double> _y;
		std::vector<svm_node*> _px;
		std::vector<svm_node> _x;
	public:
		// x is r x c row-major array of node values
		problem(int r = 0, const double* y = nullptr, int c = 0, const double* x = nullptr)
			: r(r), c(c), _y(std::vector<double>(y, y + r)), _px(r), _x(r* c)
		{
			if (x) {
				for (int i = 0; i < r; ++i) {
					for (int j = 0; j < c; ++j) {
						_x[i * c + j].index = j + 1;
						_x[i * c + j].value = x[i * c + j];
					}
					_px[i] = &_x[i * c];
				}
			}
			svm_problem::l = r;
			svm_problem::x = _px.data();
			svm_problem::y = _y.data();
		}
		problem(const problem&) = delete;
		problem& operator=(const problem&) = delete;
		~problem()
		{ }
		int rows() const
		{
			return r;
		}
		int columns() const
		{
			return c;
		}
	};

	class parameter : public svm_parameter {
	public:
		parameter(int svm_type = C_SVC, int kernel_type = RBF, int degree = 3, double gamma = 0, double coef0 = 0,
			double cache_size = 100, double eps = 1e-3, double C = 1, int nr_weight = 0, int* weight_lable = nullptr,
			double* weight = nullptr, double nu = 0.5, double p = 0.1, int shrinking = 1, int probability = 0)
			: svm_parameter{ svm_type, kernel_type, degree, gamma, coef0, cache_size, eps, C, nr_weight, weight_lable,
				weight, nu, p, shrinking, probability }
		{
		}
	};

	class model {
	public:
		svm_model* pm = nullptr;
		model(const problem& prob, const parameter& param)
		{
			const char* err = svm_check_parameter(&prob, &param);
			if (err) {
				throw std::runtime_error(err);
			}
			pm = svm_train(&prob, &param);
		}
		model(const model&) = delete;
		model& operator=(const model&) = delete;
		~model()
		{
			if (pm && pm->free_sv) {
				svm_free_and_destroy_model(&pm);
			}
		}

		svm_model& ptr() const
		{
			return *pm;
		}

	};

} // namespace fms::svm
