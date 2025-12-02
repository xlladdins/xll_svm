// fms_svm.h: Header file for Support Vector Machine (SVM) functions in the FMS library.
#pragma once
#include "svm.h"

namespace fms::svm {

	enum class svm_type { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };
	enum class kernel_type { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };

	class parameter : public svm_parameter {
	public:
		parameter(enum svm_type type = svm_type::C_SVC)
			: svm_parameter::svm_type(C_SVC)
		{
			svm_type = C_SVC;
			kernel_type = RBF;
			degree = 3;
			gamma = 0;	// 1/num_features
			coef0 = 0;
			cache_size = 100; // in MB
			eps = 1e-3;
			C = 1;
			nr_weight = 0;
			weight_label = nullptr;
			weight = nullptr;
			nu = 0.5;
			p = 0.1;
			shrinking = 1;
			probability = 0;
		}
	};

} // namespace fms::svm
