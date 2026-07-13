#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Grid = py::array_t<int, py::array::c_style | py::array::forcecast>;

struct Node {
    double f;
    double g;
    int id;
    bool operator<(const Node& other) const { return f > other.f; }
};

py::tuple astar(Grid grid_, py::tuple start, py::tuple goal, bool diagonal) {
    auto gb = grid_.request();
    if (gb.ndim != 2)
        throw std::invalid_argument("grid must be a 2D array");
    if (start.size() != 2 || goal.size() != 2)
        throw std::invalid_argument("start and goal must be 2D coordinates");

    const int rows = static_cast<int>(gb.shape[0]);
    const int cols = static_cast<int>(gb.shape[1]);
    const int sr = start[0].cast<int>();
    const int sc = start[1].cast<int>();
    const int gr = goal[0].cast<int>();
    const int gc = goal[1].cast<int>();
    auto inside = [rows, cols](int r, int c) { return r >= 0 && r < rows && c >= 0 && c < cols; };
    if (!inside(sr, sc) || !inside(gr, gc))
        throw std::invalid_argument("start and goal must be inside grid");

    const auto* cells = static_cast<const int*>(gb.ptr);
    const int start_id = sr * cols + sc;
    const int goal_id = gr * cols + gc;
    if (cells[start_id] != 0 || cells[goal_id] != 0)
        throw std::invalid_argument("start and goal must be free cells");

    const double inf = std::numeric_limits<double>::infinity();
    const double root2 = std::sqrt(2.0);
    auto heuristic = [gr, gc, diagonal, root2](int r, int c) {
        const double dr = std::abs(gr - r);
        const double dc = std::abs(gc - c);
        return diagonal ? std::max(dr, dc) + (root2 - 1.0) * std::min(dr, dc) : dr + dc;
    };

    std::vector<double> dist(rows * cols, inf);
    std::vector<int> parent(rows * cols, -2);
    std::priority_queue<Node> heap;
    dist[start_id] = 0.0;
    parent[start_id] = -1;
    heap.push({heuristic(sr, sc), 0.0, start_id});

    const int base_steps[8][3] = {
        {-1, 0, 1}, {1, 0, 1}, {0, -1, 1}, {0, 1, 1},
        {-1, -1, 2}, {-1, 1, 2}, {1, -1, 2}, {1, 1, 2},
    };
    bool found = false;
    double final_cost = inf;
    std::vector<int> ids;
    {
        py::gil_scoped_release release;
        while (!heap.empty()) {
            Node cur = heap.top();
            heap.pop();
            if (cur.g > dist[cur.id])
                continue;
            if (cur.id == goal_id) {
                found = true;
                final_cost = cur.g;
                break;
            }
            const int r = cur.id / cols;
            const int c = cur.id % cols;
            const int count = diagonal ? 8 : 4;
            for (int k = 0; k < count; ++k) {
                const int dr = base_steps[k][0];
                const int dc = base_steps[k][1];
                const int nr = r + dr;
                const int nc = c + dc;
                if (!inside(nr, nc) || cells[nr * cols + nc] != 0)
                    continue;
                if (dr != 0 && dc != 0 && (cells[(r + dr) * cols + c] != 0 || cells[r * cols + c + dc] != 0))
                    continue;
                const double step = base_steps[k][2] == 2 ? root2 : 1.0;
                const int next_id = nr * cols + nc;
                const double next_g = cur.g + step;
                if (next_g < dist[next_id]) {
                    dist[next_id] = next_g;
                    parent[next_id] = cur.id;
                    heap.push({next_g + heuristic(nr, nc), next_g, next_id});
                }
            }
        }
        if (found) {
            int id = goal_id;
            while (id != -1) {
                ids.push_back(id);
                id = parent[id];
            }
            std::reverse(ids.begin(), ids.end());
        }
    }

    if (!found)
        return py::make_tuple(py::none(), inf);
    py::list path;
    for (int id : ids)
        path.append(py::make_tuple(id / cols, id % cols));
    return py::make_tuple(path, final_cost);
}

PYBIND11_MODULE(_grid_cpp, m) {
    m.doc() = "C++ accelerated grid A* pathfinding";
    m.def("astar", &astar, py::arg("grid"), py::arg("start"), py::arg("goal"), py::arg("diagonal"));
}
