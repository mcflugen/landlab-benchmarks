import numpy as np
from landlab.graph.quantity.ext.of_link import calc_midpoint_of_link
from landlab.graph.structured_quad.ext.at_cell import fill_node_at_cell
from landlab.graph.structured_quad.ext.at_face import fill_nodes_at_face
from landlab.graph.structured_quad.ext.at_link import fill_nodes_at_link
from landlab.graph.structured_quad.ext.at_link import fill_patches_at_link
from landlab.graph.structured_quad.ext.at_node import fill_link_dirs_at_node
from landlab.graph.structured_quad.ext.at_node import fill_links_at_node
from landlab.graph.structured_quad.ext.at_node import fill_patches_at_node
from landlab.graph.structured_quad.ext.at_patch import fill_links_at_patch
from landlab.graph.structured_quad.structured_quad import UniformRectilinearGraph
from landlab.grid.divergence import calc_flux_div_at_node as calc_flux_div_at_node_slow
from landlab.grid.gradients import calc_diff_at_link as calc_diff_at_link_slow
from landlab.grid.gradients import calc_grad_at_link as calc_grad_at_link_slow
from landlab.grid.raster_divergence import calc_flux_div_at_node
from landlab.grid.raster_gradients import calc_diff_at_link
from landlab.grid.raster_gradients import calc_grad_at_link

from landlab import RasterModelGrid


def number_of_nodes(shape):
    return shape[0] * shape[1]


def number_of_links(shape):
    return shape[0] * (shape[1] - 1) + (shape[0] - 1) * shape[1]


def number_of_patches(shape):
    return (shape[0] - 1) * (shape[1] - 1)


def number_of_faces(shape):
    return number_of_links((shape[0] - 1, shape[1] - 1))


def number_of_cells(shape):
    return (shape[0] - 2) * (shape[1] - 2)


class TimeGraphQuantity:
    params = [16, 32, 64, 128, 256, 512, 1024, 2048]

    def setup(self, n):
        graph = UniformRectilinearGraph((n, n))

        self._nodes_at_link = np.empty((number_of_links((n, n)), 2), dtype=int)
        fill_nodes_at_link((n, n), self._nodes_at_link)
        self._x_of_node = graph.x_of_node
        self._y_of_node = graph.y_of_node

    def time_xy_of_link(self, n):
        out = np.empty((number_of_links((n, n)), 2), dtype=float)
        calc_midpoint_of_link(
            self._nodes_at_link,
            self._x_of_node,
            self._y_of_node,
            out,
        )


class TimeRasterFill:
    params = [16, 32, 64, 128, 256, 512, 1024, 2048]

    def time_node_at_cell(self, n):
        out = np.empty(number_of_cells((n, n)), dtype=int)
        fill_node_at_cell((n, n), out)

    def time_nodes_at_face(self, n):
        out = np.empty((number_of_faces((n, n)), 2), dtype=int)
        fill_nodes_at_face((n, n), out)

    def time_patches_at_link(self, n):
        out = np.empty((number_of_links((n, n)), 2), dtype=int)
        fill_patches_at_link((n, n), out)

    def time_nodes_at_link(self, n):
        out = np.empty((number_of_links((n, n)), 2), dtype=int)
        fill_nodes_at_link((n, n), out)

    def time_patches_at_node(self, n):
        out = np.empty((number_of_nodes((n, n)), 4), dtype=int)
        fill_patches_at_node((n, n), out)

    def time_links_at_node(self, n):
        out = np.empty((number_of_nodes((n, n)), 4), dtype=int)
        fill_links_at_node((n, n), out)

    def time_link_dirs_at_node(self, n):
        out = np.empty((number_of_nodes((n, n)), 4), dtype=np.int8)
        fill_link_dirs_at_node((n, n), out)

    def time_links_at_patch(self, n):
        out = np.empty((number_of_patches((n, n)), 4), dtype=int)
        fill_links_at_patch((n, n), out)


class TimeCalc:
    params = [[], []]
    param_names = ["module", "function"]

    func = {
        "gradients": {
            "calc_diff_at_link": calc_diff_at_link_slow,
            "calc_grad_at_link": calc_grad_at_link_slow,
            "calc_flux_div_at_node": calc_flux_div_at_node_slow,
        },
        "raster": {
            "calc_diff_at_link": calc_diff_at_link,
            "calc_flux_div_at_node": calc_flux_div_at_node,
            "calc_grad_at_link": calc_grad_at_link,
        },
    }

    def time_calculation(self, mod_name, func_name):
        self.func[mod_name][func_name](self.grid, self.value_at_node, out=self.out)


class TimeCalcAtLink(TimeCalc):
    params = [["gradients", "raster"], ["calc_diff_at_link", "calc_grad_at_link"]]

    def setup(self, mod_name, func_name):
        self.grid = RasterModelGrid((400, 5000), (1.0, 2.0))
        self.value_at_node = np.random.uniform(size=self.grid.number_of_links)
        self.out = self.grid.empty(at="link")


class TimeCalcAtNode(TimeCalc):
    params = [["gradients", "raster"], ["calc_flux_div_at_node"]]

    def setup(self, mod_name, func_name):
        self.grid = RasterModelGrid((400, 5000), (1.0, 2.0))
        self.value_at_node = np.random.uniform(size=self.grid.number_of_links)
        self.out = self.grid.empty(at="node")
