#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

struct Node {
    int feature = -1;
    int bin = -1;
    int left = -1;
    int right = -1;
    double value = 0.0;
};

class Builder {
public:
    Builder(const std::int32_t* X, const double* g, const double* h,
            int n_features, int min_leaf, int max_depth, double l2)
        : X_(X), g_(g), h_(h), n_features_(n_features), min_leaf_(min_leaf),
          max_depth_(max_depth), l2_(l2) {}

    int grow(const std::vector<int>& indices, int depth) {
        const int node_id = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{});

        double g_sum = 0.0;
        double h_sum = 0.0;
        for (int index : indices) {
            g_sum += g_[index];
            h_sum += h_[index];
        }
        h_sum = std::max(h_sum, 1e-12);

        if (indices.empty() || depth >= max_depth_ ||
            static_cast<int>(indices.size()) < 2 * min_leaf_) {
            nodes_[node_id].value = indices.empty() ? 0.0 : -g_sum / (h_sum + l2_);
            return node_id;
        }

        double best_gain = -std::numeric_limits<double>::max();
        int best_feature = -1;
        int best_bin = -1;

        for (int feature = 0; feature < n_features_; ++feature) {
            int max_bin = 0;
            for (int index : indices) {
                max_bin = std::max(max_bin, static_cast<int>(X_[index * n_features_ + feature]) + 1);
            }
            if (max_bin <= 1) {
                continue;
            }

            std::vector<double> hist_g(max_bin, 0.0);
            std::vector<double> hist_h(max_bin, 0.0);
            for (int index : indices) {
                const int bin = static_cast<int>(X_[index * n_features_ + feature]);
                hist_g[bin] += g_[index];
                hist_h[bin] += h_[index];
            }

            double left_g = 0.0;
            double left_h = 0.0;
            for (int bin = 0; bin < max_bin - 1; ++bin) {
                left_g += hist_g[bin];
                left_h += hist_h[bin];
                const double right_g = g_sum - left_g;
                const double right_h = h_sum - left_h;
                if (left_h < min_leaf_ || right_h < min_leaf_) {
                    continue;
                }
                const double gain =
                    (left_g * left_g) / (left_h + l2_) +
                    (right_g * right_g) / (right_h + l2_) -
                    (g_sum * g_sum) / (h_sum + l2_);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = feature;
                    best_bin = bin;
                }
            }
        }

        if (best_feature < 0) {
            nodes_[node_id].value = -g_sum / (h_sum + l2_);
            return node_id;
        }

        std::vector<int> left_indices;
        std::vector<int> right_indices;
        left_indices.reserve(indices.size());
        right_indices.reserve(indices.size());
        for (int index : indices) {
            if (X_[index * n_features_ + best_feature] <= best_bin) {
                left_indices.push_back(index);
            } else {
                right_indices.push_back(index);
            }
        }

        if (left_indices.empty() || right_indices.empty()) {
            nodes_[node_id].value = -g_sum / (h_sum + l2_);
            return node_id;
        }

        nodes_[node_id].feature = best_feature;
        nodes_[node_id].bin = best_bin;
        nodes_[node_id].left = grow(left_indices, depth + 1);
        nodes_[node_id].right = grow(right_indices, depth + 1);
        return node_id;
    }

    const std::vector<Node>& nodes() const { return nodes_; }

private:
    const std::int32_t* X_;
    const double* g_;
    const double* h_;
    int n_features_;
    int min_leaf_;
    int max_depth_;
    double l2_;
    std::vector<Node> nodes_;
};

void check_tree_inputs(const py::buffer_info& X, const py::buffer_info& g,
                       const py::buffer_info& h, int min_leaf, int max_depth,
                       double l2) {
    if (X.ndim != 2 || g.ndim != 1 || h.ndim != 1) {
        throw std::invalid_argument("X must be 2D and g/h must be 1D");
    }
    if (X.shape[0] != g.shape[0] || X.shape[0] != h.shape[0]) {
        throw std::invalid_argument("X, g, and h must contain the same number of rows");
    }
    if (X.shape[0] == 0 || X.shape[1] == 0) {
        throw std::invalid_argument("X must be non-empty");
    }
    if (min_leaf < 1 || max_depth < 0 || l2 < 0.0) {
        throw std::invalid_argument("min_leaf, max_depth, and l2 must be valid");
    }
}

py::tuple fit_hist_tree(
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> g,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    int min_leaf, int max_depth, double l2) {
    const auto Xb = X.request();
    const auto gb = g.request();
    const auto hb = h.request();
    check_tree_inputs(Xb, gb, hb, min_leaf, max_depth, l2);

    Builder builder(
        static_cast<const std::int32_t*>(Xb.ptr),
        static_cast<const double*>(gb.ptr),
        static_cast<const double*>(hb.ptr),
        static_cast<int>(Xb.shape[1]), min_leaf, max_depth, l2);
    std::vector<int> indices(static_cast<std::size_t>(Xb.shape[0]));
    for (int i = 0; i < Xb.shape[0]; ++i) {
        indices[static_cast<std::size_t>(i)] = i;
    }
    {
        py::gil_scoped_release release;
        builder.grow(indices, 0);
    }

    const auto& nodes = builder.nodes();
    py::array_t<std::int32_t> features(nodes.size());
    py::array_t<std::int32_t> bins(nodes.size());
    py::array_t<std::int32_t> left(nodes.size());
    py::array_t<std::int32_t> right(nodes.size());
    py::array_t<double> values(nodes.size());
    auto fv = features.mutable_unchecked<1>();
    auto bv = bins.mutable_unchecked<1>();
    auto lv = left.mutable_unchecked<1>();
    auto rv = right.mutable_unchecked<1>();
    auto vv = values.mutable_unchecked<1>();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        fv(i) = nodes[i].feature;
        bv(i) = nodes[i].bin;
        lv(i) = nodes[i].left;
        rv(i) = nodes[i].right;
        vv(i) = nodes[i].value;
    }
    return py::make_tuple(features, bins, left, right, values);
}

py::array_t<double> predict_hist_tree(
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> X,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> features,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> bins,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> left,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> right,
    py::array_t<double, py::array::c_style | py::array::forcecast> values) {
    const auto Xb = X.request();
    const auto fb = features.request();
    const auto bb = bins.request();
    const auto lb = left.request();
    const auto rb = right.request();
    const auto vb = values.request();
    if (Xb.ndim != 2 || fb.ndim != 1 || bb.ndim != 1 || lb.ndim != 1 ||
        rb.ndim != 1 || vb.ndim != 1 || fb.shape[0] == 0 ||
        fb.shape[0] != bb.shape[0] || fb.shape[0] != lb.shape[0] ||
        fb.shape[0] != rb.shape[0] || fb.shape[0] != vb.shape[0]) {
        throw std::invalid_argument("invalid histogram tree arrays");
    }

    auto* x_ptr = static_cast<const std::int32_t*>(Xb.ptr);
    auto* f_ptr = static_cast<const std::int32_t*>(fb.ptr);
    auto* b_ptr = static_cast<const std::int32_t*>(bb.ptr);
    auto* l_ptr = static_cast<const std::int32_t*>(lb.ptr);
    auto* r_ptr = static_cast<const std::int32_t*>(rb.ptr);
    auto* v_ptr = static_cast<const double*>(vb.ptr);
    const int n = static_cast<int>(Xb.shape[0]);
    const int d = static_cast<int>(Xb.shape[1]);
    const int n_nodes = static_cast<int>(fb.shape[0]);
    py::array_t<double> out(n);
    auto* out_ptr = static_cast<double*>(out.request().ptr);

    {
        py::gil_scoped_release release;
        for (int i = 0; i < n; ++i) {
            int node = 0;
            while (f_ptr[node] >= 0) {
                if (f_ptr[node] >= d) {
                    throw std::invalid_argument("tree feature index is out of range");
                }
                node = x_ptr[i * d + f_ptr[node]] <= b_ptr[node]
                    ? l_ptr[node] : r_ptr[node];
                if (node < 0 || node >= n_nodes) {
                    throw std::invalid_argument("tree child index is out of range");
                }
            }
            out_ptr[i] = v_ptr[node];
        }
    }
    return out;
}

PYBIND11_MODULE(_hist_cpp, m) {
    m.doc() = "C++ histogram tree kernels";
    m.def("fit_hist_tree", &fit_hist_tree, "Fit one histogram tree");
    m.def("predict_hist_tree", &predict_hist_tree, "Predict one histogram tree");
}
