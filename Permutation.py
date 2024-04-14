import numpy as np
from math import prod


class Permutation:
    """
    Class representing a permutation of enumerated nodes in cubic lattice.

    ----------
    perm - 1D np array representing the permutation of nodes
    cube_dimensions - tuple containing dimensions of cubic lattice. Sometimes consecutive dimensions are referred to as m1, m2 , ...
    size - number of elements if the lattice
    """

    def __init__(self, perm: np.array, cube_dimensions: tuple):
        self.perm = perm
        self.cube_dimensions = cube_dimensions

    @staticmethod
    def __invert_permutation(permutation: np.array) -> np.array:
        inv = np.empty_like(permutation)
        inv[permutation] = np.arange(len(inv), dtype=int)
        return inv

    def invert(self):
        return Permutation(Permutation.__invert_permutation(self.perm), self.cube_dimensions)

    @staticmethod
    def __get_cycles(n: int, perm: np.array) -> list:
        """ decompose permutation into cycles """
        visited_tab = [False] * n
        cycle_tab = []
        for ind, el in enumerate(perm):
            if visited_tab[ind]:
                continue
            new_cycle = []
            current_ind = ind
            while not visited_tab[current_ind]:
                new_cycle.append(current_ind)
                visited_tab[current_ind] = True
                current_ind = perm[current_ind]
            cycle_tab.append(new_cycle)
        return cycle_tab

    @staticmethod
    def __decompose_cycle_to_swaps(k: int, cycle: list) -> np.array:
        """ return the decomposition of cycle of k nodes into k-1 swaps between arbitrary elements """
        swap_array = np.zeros((k - 1, 2), dtype=int)
        for i in range(k // 2):
            swap_array[i, 0] = cycle[-i]
            swap_array[i, 1] = cycle[1 + i]
        for i in range(k - (k // 2) - 1):
            swap_array[k // 2 + i, 0] = cycle[1 + i]
            swap_array[k // 2 + i, 1] = cycle[-1 - i]
        return swap_array

    @staticmethod
    def __get_minimal_swap_decomposition(n: int, perm: np.array) -> np.array:
        """
        return the decomposition of permutation into minimal number of swaps between arbitrary elements
        """
        cycles_tab = Permutation.__get_cycles(n, perm)
        swap_array = np.zeros((n - len(cycles_tab), 2), dtype=int)
        ind = 0
        for cycle in cycles_tab:
            k = len(cycle)
            swap_array[ind:ind + k - 1] = Permutation.__decompose_cycle_to_swaps(k, cycle)
            ind += k - 1
        return swap_array

    @staticmethod
    def __apply_swap_1D(swap, perm):
        perm[swap[1]], perm[swap[0]] = perm[swap[0]], perm[swap[1]]
        return perm

    @staticmethod
    def __bare_brick_sort(m: int, perm0: np.array, start_shift: int = 0) -> tuple[int, list]:
        """
        Decomposes the permutation perm0 of m elements in to neighbour swaps by brick sort (also known as left-right sort, parallel bubble sort)

        Parameters
        ----------
        perm0 - permutation of interest
        m - number of elements in permutation
        start_shift - shift of first pair in consideration, equal 0 or 1

        Returns
        ----------
        layer_counter - number of layers in permutation implementation
        swap_list1 - list of consecutive swaps to implement a permutation. Each swap encoded as a pair of indexes of swapped nodes
        """
        perm = perm0.copy()
        layer_counter = 0
        was_sorted1 = False
        was_sorted2 = False
        swap_list = []

        while True:
            if start_shift == 0:
                was_sorted1 = True
                for i in range(m // 2):
                    if perm[2 * i] > perm[2 * i + 1]:
                        perm[2 * i], perm[2 * i + 1] = perm[2 * i + 1], perm[2 * i]
                        was_sorted1 = False
                        swap_list.append([2 * i, 2 * i + 1])

                if was_sorted1 and was_sorted2:
                    break
                elif not was_sorted1:
                    layer_counter += 1

            was_sorted2 = True
            for i in range(m // 2 - 1 + m % 2):
                if perm[2 * i + 1] > perm[2 * i + 2]:
                    perm[2 * i + 1], perm[2 * i + 2] = perm[2 * i + 2], perm[2 * i + 1]
                    was_sorted2 = False
                    swap_list.append([2 * i + 1, 2 * i + 2])

            if was_sorted1 and was_sorted2:
                break
            elif not was_sorted2:
                layer_counter += 1
            start_shift = 0

        return layer_counter, swap_list[::-1]

    @staticmethod
    def __brick_sort1D(m: int, perm: np.array) -> tuple[int, list]:
        """
        Decomposes the permutation perm of m elements in to neighbour swaps by brick sort (also knows as left-right sort, parallel bubble sort).
        Checks two possible starting positions for find the better one in terms of necessary number of layers.

        Parameters
        ----------
        perm - permutation of interest
        m - number of elements in permutation

        Returns
        ----------
        layer_n1/layer_n2 - number of layers in permutation implementation
        swap_list1/swap_list2 - list of consecutive swaps to implement a permutation. Each swap encoded as pair of indexes of swapped nodes
        """
        layer_n1, swap_list1 = Permutation.__bare_brick_sort(m, perm, start_shift=0)
        layer_n2, swap_list2 = Permutation.__bare_brick_sort(m, perm, start_shift=1)

        # the number of swaps if always the same, but the number of layers might depend on which layer one starts
        if layer_n1 < layer_n2:
            return layer_n1, swap_list1
        else:
            return layer_n2, swap_list2

    @staticmethod
    def __enumerate_permutation_elements(m1: int, m2: int, perm: np.array) -> np.array:
        """
        Creates a correct enumeration of permutation elements for the first sort for teh rectangle lattice of dimensions

        Parameters
        ----------
        m1 - first dimension of rectangle
        m2 - second dimension of rectangle
        perm - permutation of interest

        Returns
        ----------
        enumeration - 1D array with enumeration of rectangle elements
        """
        swap_decomposition = Permutation.__get_minimal_swap_decomposition(m1 * m2, perm)
        perm0 = np.linspace(0, m1 * m2 - 1, m1 * m2).astype(int)
        enumeration = perm0.reshape(m1, m2) // m2

        for swap in swap_decomposition:
            perm0 = Permutation.__apply_swap_1D(swap, perm0)
            original_column_ind = perm0.reshape(m1, m2) % m2

            x1 = swap[0] // m2
            y1 = swap[0] % m2
            x2 = swap[1] // m2
            y2 = swap[1] % m2
            if enumeration[x1, y1] == enumeration[x2, y2] or original_column_ind[x1, y1] == original_column_ind[x2, y2]:
                continue
            if y1 == y2:
                enumeration[x1, y1], enumeration[x2, y2] = enumeration[x2, y2], enumeration[x1, y1]
            else:

                y1_new = y1
                x2_new = x2
                y2_new = y2
                x2_old = x1

                to_visit = np.full(m2, True)
                while to_visit[y2_new]:
                    # exchange the elements with the same numbers in second column
                    x1_new = np.where(enumeration[:, y2_new] == enumeration[x2_old, y1_new])[0][0]
                    y1_new = y2_new
                    enumeration[x1_new, y1_new], enumeration[x2_new, y2_new] = enumeration[x2_new, y2_new], enumeration[
                        x1_new, y1_new]
                    to_visit[y1_new] = False

                    # find new elements in new column to exchange
                    new_el = np.where((original_column_ind == original_column_ind[x1_new, y1_new]) &
                                      (enumeration == enumeration[x1_new, y1_new]))
                    x2_old = x2_new
                    if to_visit[new_el[1][0]]:
                        x2_new = new_el[0][0]
                        y2_new = new_el[1][0]
                    elif len(new_el[1]) == 2:  # in new_el are at most two points
                        x2_new = new_el[0][1]
                        y2_new = new_el[1][1]
                    else:
                        break

        return enumeration

    def get_enumeration_first_dimension(self) -> np.array:
        """
        Returns a correct enumeration of elements for the first sort, after flattening the lattice into a
        two-dimensional. In each column there will be indexes form 0 to m1-1 and elements which should end in column
        will be indexed form 0 to m1-1.

        Returns
        ----------
        1D array with enumeration of rectangle elements
        """
        m1 = self.cube_dimensions[0]
        m2 = np.prod(self.cube_dimensions[1:])
        return self.__enumerate_permutation_elements(m1, m2, self.perm)

    @staticmethod
    def __check_enumeration(m1: int, m2: int, perm: np.array, enumeration: np.array) -> bool:
        """
        Used for testing
        Check whether enumeration of permutation elements form enumerate_rectangle_permutation_elements is correct.

        Parameters
        ----------
        m1 - first dimension of rectangle
        m2 - second dimension of rectangle
        perm - permutation of interest
        enumeration - enumeration to check
        """

        ordered_sequence = np.linspace(0, m1 - 1, m1).astype(int)
        for i in range(m2):
            sorted_col = enumeration[:, i].copy()
            sorted_col.sort()
            if np.any(sorted_col != ordered_sequence):
                return False

        original_column_ind = perm.reshape(m1, m2) % m2
        for i in range(m2):
            sorted_sequence = enumeration[original_column_ind == i].copy()
            sorted_sequence.sort()
            if np.any(sorted_sequence != ordered_sequence):
                return False
        return True

    @staticmethod
    def __hypercube_sort(cube_dimensions: tuple, perm0: np.array) -> tuple[int, list]:
        """
        Recursively decomposes the permutation on hypercube lattice into nearest neighbour swaps.
        The complexity of solution is optimal with respect to both number of swaps and layers in decomposition

        Parameters
        ----------
        cube_dimensions - dimensions of hypercube
        perm0 - permutation to decompose

        Returns
        ----------
        layer_counter - number of layers in permutation implementation
        swap_list - list of consecutive swaps to implement a permutation. Each swap encoded as pair of indexes of swapped nodes
        """

        swap_list = []
        layer_counter = 0
        perm = perm0.copy()

        if len(cube_dimensions) == 1:
            return Permutation.__brick_sort1D(cube_dimensions[0], perm)
        else:
            m1 = cube_dimensions[0]
            m2 = prod(cube_dimensions[1:])

            enumeration = Permutation.__enumerate_permutation_elements(m1, m2, perm)
            # if not Permutation.__check_enumeration(m1, m2, perm, enumeration):
            #    raise Exception("Wrong Enumeration")

            # sorting columns by enumeration
            lay_counter1 = 0
            for column_ind in range(m2):
                sub_lay_counter, sub_swap_list = Permutation.__brick_sort1D(m1, enumeration[:, column_ind])
                if sub_lay_counter > lay_counter1:
                    lay_counter1 = sub_lay_counter

                sub_swap_list = sub_swap_list[::-1]
                for swap in sub_swap_list:
                    swap[0] = swap[0] * m2 + column_ind
                    swap[1] = swap[1] * m2 + column_ind
                swap_list.extend(sub_swap_list)

                # slightly faster way, than apply swap by swap
                indices = np.argsort(enumeration[:, column_ind])
                perm.reshape((m1, m2))[:, column_ind] = perm.reshape((m1, m2))[:, column_ind][indices]

            layer_counter += lay_counter1

            # sorting the hyper row inductively
            lay_counter2 = 0
            for i in range(m1):
                # rename elements in hyper row, so each hyper row looks like permutation to decompose
                aux_perm = perm.reshape((m1, m2))[i] % m2
                sub_lay_counter, sub_swap_list = Permutation.__hypercube_sort(cube_dimensions[1:], aux_perm)
                if sub_lay_counter > lay_counter2:
                    lay_counter2 = sub_lay_counter

                sub_swap_list = sub_swap_list[::-1]
                for swap in sub_swap_list:
                    swap[0] = swap[0] + i * m2
                    swap[1] = swap[1] + i * m2
                swap_list.extend(sub_swap_list)

                # slightly faster way, than apply swap by swap
                perm.reshape((m1, m2))[i] = (perm.reshape((m1, m2))[i])[np.argsort(aux_perm)]

            layer_counter += lay_counter2

            # sorting columns by enumeration
            lay_counter3 = 0
            for column_ind in range(m2):
                sub_lay_counter, sub_swap_list = Permutation.__brick_sort1D(m1, perm.reshape((m1, m2))[:, column_ind])
                if sub_lay_counter > lay_counter3:
                    lay_counter3 = sub_lay_counter

                sub_swap_list = sub_swap_list[::-1]
                for swap in sub_swap_list:
                    swap[0] = swap[0] * m2 + column_ind
                    swap[1] = swap[1] * m2 + column_ind
                swap_list.extend(sub_swap_list)

                # slightly faster way, than apply swap by swap
                perm.reshape((m1, m2))[:, column_ind].sort()

            layer_counter += lay_counter3

            return layer_counter, swap_list[::-1]

    def hypercube_sort(self):
        """
        Decomposes the permutation on hypercube lattice into nearest neighbour swaps.
        The complexity of solution is optimal with respect to both number of swaps and layers in decomposition

        Returns
        ----------
        layer_counter - number of layers in permutation implementation
        swap_list - list of consecutive swaps to implement a permutation. Each swap encoded as pair of indexes of swapped nodes
        """
        return self.__hypercube_sort(self.cube_dimensions, self.perm)

    def give_minimal_layer_num(self) -> float:
        """
        Finds the length of the longest path for element from its original position
        into a final (permuted)
        """
        paths_l = np.zeros(self.cube_dimensions)
        perm = self.perm.copy()

        for i in range(len(self.cube_dimensions)):
            coord = eval("coord.reshape((" + "1," * i + "self.cube_dimensions[" + str(i) + "]" + ",1" * (
                    len(self.cube_dimensions) - i - 1) + "))")
            perm_coord = perm.reshape(self.cube_dimensions) // prod(self.cube_dimensions[i + 1:])
            if i > 0:
                perm_coord = perm_coord % self.cube_dimensions[i - 1]
            paths_l += np.abs(perm_coord - coord)

        return np.max(paths_l)
