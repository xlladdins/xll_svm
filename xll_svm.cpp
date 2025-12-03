// xll_svm.cpp
#include "fms_svm.h"
#include "xll_svm.h"

using namespace xll;
using namespace fms::svm;

AddIn xai_svm_version(
	Function(XLL_INT, L"xll_svm_version", L"SVM.VERSION")
	.Arguments({})
	.Category(CATEGORY)
	.FunctionHelp(L"Return the LIBSVM version number.")
);
int WINAPI xll_svm_version()
{
#pragma XLLEXPORT
	return libsvm_version;
}

AddIn xai_svm_problem_(
	Function(XLL_HANDLEX, L"xll_svm_problem_", L"\\SVM.PROBLEM")
	.Arguments({
		Arg(XLL_FP, L"y", L"is the array of r target values."),
		Arg(XLL_FP, L"x", L"is the r x c row-major array of feature values."),
	})
	.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp(L"Create an SVM problem handle from the training data.")
);
HANDLEX WINAPI xll_svm_problem_(const _FP12* py, const _FP12* px)
{
#pragma XLLEXPORT
	HANDLEX h = INVALID_HANDLEX;
	try {
		ensure(size(*py) == rows(*px) || !L"\\SVM.PROBLEM: size(y) must equal rows(x)");

		handle<problem> h_(new problem(rows(*px), py->array, columns(*px), px->array));
		ensure(h_ || ! L"\\SVM.PROBLEM: could not create problem handle");

		h = h_.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR("\\SVM.PROBLEM: unknown exception");
	}
	return h;
}

AddIn xai_svm_problem(
	Function(XLL_FP, L"xll_svm_problem", L"SVM.PROBLEM")
	.Arguments({
		Arg(XLL_HANDLEX, L"h", L"is a handle from \\SVM.PROBLEM."),
	})
	.Category(CATEGORY)
	.FunctionHelp(L"Create an empty SVM problem handle.")
);
_FP12* WINAPI xll_svm_problem(HANDLEX h)
{
#pragma XLLEXPORT
	static xll::FP12 p;
	try {
		handle<problem> h_(h);
		ensure(h_ || !L"SVM.PROBLEM: invalid problem handle");
		const problem& h = *h_;

		int r = h.rows();
		int c = h.columns();
		p.resize(r + 1, c + 1);
		p(0, 0) = 0; // unused
		// indices in first row
		for (int j = 1; j <= c; ++j) {
			p(0, j) = j;
		}
		// labels in first column
		for (int i = 0; i < r; ++i) {
			p(i + 1, 0) = h.y[i];
			for (int j = 0; j < c; ++j) {
				p(i + 1, j + 1) = h.x[i][j].value;
			}
		}
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR("xll_svm_problem: unknown exception");
	}

	return p.get();
}

AddIn xai_svm_parameter_(
	Function(XLL_HANDLEX, L"xll_svm_parameter_", L"\\SVM.PARAMETER")
	.Arguments({
		Arg(XLL_INT, L"svm_type", L"is the SVM type."),
		Arg(XLL_INT, L"kernel_type", L"is the kernel type."),
		Arg(XLL_INT, L"degree", L"is the degree for the polynomial kernel."),
		Arg(XLL_DOUBLE, L"gamma", L"is the gamma for the polynomial/rbf/sigmoid kernel."),
		Arg(XLL_DOUBLE, L"coef0", L"is the coef0 for the polynomial/sigmoid kernel."),
		Arg(XLL_DOUBLE, L"cache_size", L"is the cache size in MB."),
		Arg(XLL_DOUBLE, L"eps", L"is the stopping criteria."),
		Arg(XLL_DOUBLE, L"C", L"is the C parameter for C_SVC, EPSILON_SVR and NU_SVR."),
		Arg(XLL_INT, L"nr_weight", L"is the number of weights for C_SVC."),
	})
	.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp(L"Create an SVM parameter handle.")
);
HANDLEX WINAPI xll_svm_parameter_(int svm_type, int kernel_type, int degree, double gamma, double coef0,
	double cache_size, double eps, double C, int nr_weight/*, int* weight_label, double* weight, double nu,
	double p, int shrinking, int probability*/)
{
#pragma XLLEXPORT
	HANDLEX h = INVALID_HANDLEX;
	try {
		handle<parameter> h_(new parameter(/*svm_type, kernel_type, degree, gamma, coef0, cache_size, eps, C,
			nr_weight, weight_label, weight, nu, p, shrinking, probability*/));
		ensure(h_ || !L"\\SVM.PARAMETER: could not create parameter handle");
		h = h_.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR("xll_svm_parameter: unknown exception");
	}
	return h;
}

AddIn xai_svm_parameter(
	Function(XLL_LPOPER, L"xll_svm_parameter", L"SVM.PARAMETER")
	.Arguments({
		Arg(XLL_HANDLEX, L"h", L"is a handle from \\SVM.PARAMETER."),
	})
	.Category(CATEGORY)
	.FunctionHelp(L"Return the SVM parameter values.")
);
LPXLOPER12 WINAPI xll_svm_parameter(HANDLEX h)
{
#pragma XLLEXPORT
	static OPER p(2, 9);
	try {
		handle<parameter> h_(h);
		ensure(h_ || !L"SVM.PARAMETER: invalid parameter handle");
		const parameter& h = *h_;

		p(0, 0) = L"sum_type";    p(1, 0) = h.svm_type;
		p(0, 1) = L"kernel_type"; p(1, 1) = h.kernel_type;
		p(0, 2) = L"degree";      p(1, 2) = h.degree;
		p(0, 3) = L"gamma";       p(1, 3) = h.gamma;
		p(0, 4) = L"coef0";       p(1, 4) = h.coef0;
		p(0, 5) = L"cache_size";  p(1, 5) = h.cache_size;
		p(0, 6) = L"eps";         p(1, 6) = h.eps;
		p(0, 7) = L"C";           p(1, 7) = h.C;
		p(0, 8) = L"nr_weight";   p(1, 8) = h.nr_weight;
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR("xll_svm_parameter: unknown exception");
	}

	return &p;
}

AddIn xai_svm_model_(
	Function(XLL_HANDLEX, L"xll_svm_model_", L"\\SVM.MODEL")
	.Arguments({
		Arg(XLL_HANDLEX, L"problem", L"is a handle from \\SVM.PROBLEM."),
		Arg(XLL_HANDLEX, L"parameter", L"is a handle from \\SVM.PARAMETER."),
		})
		.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp(L"Create an SVM model handle by training the SVM model.")
);
HANDLEX WINAPI xll_svm_model_(HANDLEX h_problem, HANDLEX h_parameter)
{	
#pragma	XLLEXPORT
	HANDLEX h = INVALID_HANDLEX;
	try {
		handle<problem> h_prob(h_problem);
		ensure(h_prob || !L"\\SVM.MODEL: invalid problem handle");

		handle<parameter> h_param(h_parameter);
		ensure(h_param || !L"\\SVM.MODEL: invalid parameter handle");
		
		handle<model> h_(new model(*h_prob, *h_param));
		ensure(h_ || !L"\\SVM.MODEL: could not create model handle");
		
		h = h_.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR("\\SVM.MODEL: unknown exception");
	}
	return h;
}

AddIn xai_svm_model(
	Function(XLL_LPOPER, L"xll_svm_model", L"SVM.MODEL")
	.Arguments({
		Arg(XLL_HANDLEX, L"h", L"is a handle from \\SVM.MODEL."),
	})
	.Category(CATEGORY)
	.FunctionHelp(L"Return the support vectors and coefficients from the SVM model.")
);	
LPXLOPER12 WINAPI xll_svm_model(HANDLEX h)
{
#pragma XLLEXPORT
	static OPER m(2, 4);
	try {
		handle<model> h_(h);
		ensure(h_ || !L"SVM.MODEL: invalid model handle");

		const model& h = *h_;

		m(0, 0) = L"svm_type"; m(1, 0) = h.pm->param.svm_type;
		m(0, 1) = L"kernel_type"; m(1, 1) = h.pm->param.kernel_type;
		// kernet_type POLY ...
		m(0, 2) = L"nr_class"; m(1, 2) = h.pm->nr_class;
		m(0, 3) = L"nr_sv"; m(1, 3) = h.pm->l;
		/*
		int n_sv = h.get_nr_sv();
		int n_features = h.get_nr_features();
		m.resize(n_sv + 1, n_features + 2);
		m(0, 0) = 0; // unused
		// indices in first row
		for (int j = 1; j <= n_features; ++j) {
			m(0, j) = j;
		}
		m(0, n_features + 1) = L"alpha";
		// support vectors and coefficients
		for (int i = 0; i < n_sv; ++i) {
			const svm_node* sv = h.get_sv(i);
			for (int j = 0; j < n_features; ++j) {
				m(i + 1, j + 1) = sv[j].value;
			}
			m(i + 1, n_features + 1) = h.get_alpha(i);
		}
		*/
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR("xll_svm_model: unknown exception");
	}
	return &m;
}