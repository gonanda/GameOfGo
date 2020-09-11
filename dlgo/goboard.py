import copy
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result
from dlgo import zobrist


neighbor_tables = {}
corner_tables = {}


def init_neighbor_table(dim):
    rows, cols = dim
    new_table = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            p = Point(row=r, col=c)
            if r == 1:
                if c == 1:
                    neighbor_list = [
                        Point(row=1, col=2),
                        Point(row=2, col=1),
                    ]
                elif c == cols:
                    neighbor_list = [
                        Point(row=2, col=cols),
                        Point(row=1, col=cols-1),
                    ]
                else:
                    neighbor_list = [
                        Point(row=1, col=c+1),
                        Point(row=2, col=c),
                        Point(row=1, col=c-1),
                    ]
            elif r == rows:
                if c == 1:
                    neighbor_list = [
                        Point(row=rows-1, col=1),
                        Point(row=rows, col=2),
                    ]
                elif c == cols:
                    neighbor_list = [
                        Point(row=rows, col=cols-1),
                        Point(row=rows-1, col=cols),
                    ]
                else:
                    neighbor_list = [
                        Point(row=rows, col=c-1),
                        Point(row=rows-1, col=c),
                        Point(row=rows, col=c+1),
                    ]
            else:
                if c == 1:
                    neighbor_list = [
                        Point(row=r-1, col=1),
                        Point(row=r, col=2),
                        Point(row=r+1, col=1),
                    ]
                elif c == cols:
                    neighbor_list = [
                        Point(row=r+1, col=cols),
                        Point(row=r, col=cols-1),
                        Point(row=r-1, col=cols),
                    ]
                else:
                    neighbor_list = [
                        Point(row=r, col=c+1),
                        Point(row=r+1, col=c),
                        Point(row=r, col=c-1),
                        Point(row=r-1, col=c),
                    ]
            new_table[p] = neighbor_list
    neighbor_tables[dim] = new_table


def init_corner_table(dim):
    rows, cols = dim
    new_table = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            p = Point(row=r, col=c)
            if r == 1:
                if c == 1:
                    corner_list = [
                        Point(row=2, col=2),
                    ]
                elif c == cols:
                    corner_list = [
                        Point(row=2, col=cols-1),
                    ]
                else:
                    corner_list = [
                        Point(row=2, col=c+1),
                        Point(row=2, col=c-1),
                    ]
            elif r == rows:
                if c == 1:
                    corner_list = [
                        Point(row=rows-1, col=2),
                    ]
                elif c == cols:
                    corner_list = [
                        Point(row=rows-1, col=cols-1),
                    ]
                else:
                    corner_list = [
                        Point(row=rows-1, col=c-1),
                        Point(row=rows-1, col=c+1),
                    ]
            else:
                if c == 1:
                    corner_list = [
                        Point(row=r-1, col=2),
                        Point(row=r+1, col=2),
                    ]
                elif c == cols:
                    corner_list = [
                        Point(row=r+1, col=cols-1),
                        Point(row=r-1, col=cols-1),
                    ]
                else:
                    corner_list = [
                        Point(row=r+1, col=c+1),
                        Point(row=r+1, col=c-1),
                        Point(row=r-1, col=c-1),
                        Point(row=r-1, col=c+1),
                    ]
            new_table[p] = corner_list
    corner_tables[dim] = new_table


class Move():
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)

    def transpose(self):
        if self.is_play:
            return Move(point=Point(row=self.point.col,col=self.point.row))
        else:
            return self

    def flip_row(self, board_size):
        if self.is_play:
            return Move(point=Point(row=board_size+1-self.point.row,col=self.point.col))
        else:
            return self

    def flip_col(self, board_size):
        if self.is_play:
            return Move(point=Point(row=self.point.row,col=board_size+1-self.point.col))
        else:
            return self

    def flip_row_col(self, board_size):
        if self.is_play:
            return Move(point=Point(row=board_size+1-self.point.row,col=board_size+1-self.point.col))
        else:
            return self

    def __str__(self):
        if self.is_pass:
            return 'pass'
        if self.is_resign:
            return 'resign'
        return 'play %s' % str(self.point)

    def __hash__(self):
        return hash((
            self.is_play,
            self.is_pass,
            self.is_resign,
            self.point))

    def  __eq__(self, other):
        return (
            self.is_play,
            self.is_pass,
            self.is_resign,
            self.point) == (
            other.is_play,
            other.is_pass,
            other.is_resign,
            other.point)

    __repr__ = __str__


class GoString():
    def __init__(self, color, stones):
        self.color = color
        self.stones = frozenset(stones)

    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones

    def __hash__(self):
        return(hash(self.stones))

    def __deepcopy__(self, memodict={}):
        return GoString(self.color, self.stones)

    def __str__(self):
        go_string_string = '<String '
        for stone in self.stones:
            go_string_string += (str(stone)+' ')
        go_string_string += (str(self.color)+'>')
        return go_string_string

    __repr__ = __str__


class StringConnection():
    def __init__(self, color, strings):
        self.color = color
        self.strings = frozenset(strings)

    def without_strings(self, strings):
        new_strings = self.strings - set(strings)
        return StringConnection(self.color, new_strings)

    def merged_with(self, string_connection):
        assert string_connection.color == self.color
        combined_strings = self.strings | string_connection.strings
        return StringConnection(
            self.color,
            combined_strings)

    def __eq__(self, other):
        return isinstance(other, StringConnection) and \
            self.color == other.color and \
            self.strings == other.strings

    def __hash__(self):
        return(hash(self.strings))

    def __deepcopy__(self, memodict={}):
        return StringConnection(self.color, self.strings)

    def __str__(self):
        connection_string = '[Connection '
        for string in self.strings:
            connection_string += (str(string)+' ')
        connection_string += ']'
        return connection_string

    __repr__ = __str__


class Region():
    def __init__(self, color, points):
        self.color = color
        self.points = frozenset(points)

    def __eq__(self, other):
        return isinstance(other, Region) and \
            self.color == other.color and \
            self.points == other.points

    def __hash__(self):
        return(hash(self.points))

    def __deepcopy__(self, memodict={}):
        return Region(self.color, self.points)

    def __str__(self):
        region_string = '[Region '
        for point in self.points:
            region_string += (str(point)+' ')
        region_string += str(self.color)
        region_string += ']'
        return region_string

    __repr__ = __str__


class Board():

    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        global all_points
        try: all_points
        except NameError: all_points = None
        if all_points is None:
            all_points = set()
            for r in range(1,self.num_rows+1):
                for c in range(1,self.num_cols+1):
                    all_points.add(Point(col=c,row=r))
        self.edge_strings = {Player.black: GoString(Player.black, {}),Player.white: GoString(Player.white, {})}
        self.start_regions = {Player.black: Region(Player.black,frozenset(all_points)),Player.white: Region(Player.white,frozenset(all_points))}
        self._grid = {}
        self._liberties = {}
        self._connected = {self.edge_strings[Player.black]: set(), self.edge_strings[Player.white]: set()}
        self._connections = {self.edge_strings[Player.black]: StringConnection(Player.black,[self.edge_strings[Player.black]]),
            self.edge_strings[Player.white]: StringConnection(Player.white,[self.edge_strings[Player.white]])}
        self._region_by_point_black = {p: self.start_regions[Player.black] for p in all_points}
        self._region_by_point_white = {p: self.start_regions[Player.white] for p in all_points}
        self._regions_by_string = {}
        self._strings_by_region = {}
        self._safe_strings_by_region = {Player.black: {},Player.white: {}}
        self._vital_regions_by_string = {Player.black: {},Player.white: {}}
        self._potentially_safe_strings_by_region = {Player.black: {},Player.white: {}}
        self._healthy_regions_by_string = {Player.black: {},Player.white: {}}
        self._hash = zobrist.EMPTY_BOARD


        global neighbor_tables
        dim = (num_rows, num_cols)
        if dim not in neighbor_tables:
            init_neighbor_table(dim)
        self.neighbor_table = neighbor_tables[dim]

        global corner_tables
        dim = (num_rows, num_cols)
        if dim not in corner_tables:
            init_corner_table(dim)
        self.corner_table = corner_tables[dim]

    def assign_new_region_to_point(self, color, region, point):
        if color==Player.black:
            self._region_by_point_black[point] = region
        else:
            self._region_by_point_white[point] = region

    def read_region_by_point(self, color, point):
        if color==Player.black:
            return self._region_by_point_black[point]
        else:
            return self._region_by_point_white[point]

    def delete_point_from_region_by_point(self, color, point):
        if color==Player.black:
            del(self._region_by_point_black[point])
        else:
            del(self._region_by_point_white[point])

    def neighbors(self, point):
        return self.neighbor_table[point]

    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols

    def get(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    def string_delete_liberty(self, string, point):
        self._liberties[string] = self._liberties[string]-{point}

    def string_add_liberty(self, string, point):
        self._liberties[string] = self._liberties[string]|{point}

    def string_delete_connected(self, string, connected):
        self._connected[string] = self._connected[string]-{connected}

    def string_add_connected(self, string, connected):
        self._connected[string] = self._connected[string]|{connected}

    def strings_merged(self, this_string, that_string):
        assert this_string.color == that_string.color
        combined_stones = this_string.stones | that_string.stones
        combined_liberties = (self._liberties[this_string] | self._liberties[that_string]) - combined_stones
        combined_connected = (self._connected[this_string] | self._connected[that_string])
        new_string = GoString(this_string.color, combined_stones)
        del(self._liberties[this_string])
        del(self._liberties[that_string])
        del(self._connected[this_string])
        del(self._connected[that_string])
        self._liberties[new_string] = set(combined_liberties)
        self._connected[new_string] = set(combined_connected)
        return new_string

    def num_liberties(self, string):
        return len(self._liberties[string])

    def find_connections(self, start_string, found_connections = None):
        if found_connections == None:
            found_connections = set([])
        if start_string in found_connections:
            return set()
        found_connections |= {start_string}
        for string in self._connected[start_string]:
            found_connections |= self.find_connections(string, found_connections)
        return found_connections

    def divide_region(self, player, point):

        if(len(self.neighbor_table[point])==2):
            free_neighbors = []
            corner = self.corner_table[point][0]
            for neighbor in self.neighbor_table[point]:
                if self.get(neighbor)!=player:
                    free_neighbors.append(neighbor)
            if len(free_neighbors)==0:
                return []
            elif len(free_neighbors)==1:
                return free_neighbors
            else:
                if self._connections.get(self._grid.get(corner)) is self._connections[self.edge_strings[player]]:
                    return free_neighbors
                else:
                    return [free_neighbors[0]]

        if(len(self.neighbor_table[point])==3):
            free_neighbors = []
            corners = self.corner_table[point]
            for neighbor in self.neighbor_table[point]:
                if self.get(neighbor)!=player:
                    free_neighbors.append(neighbor)

            if len(free_neighbors)==0:
                return []
            elif len(free_neighbors)==1:
                return free_neighbors
            elif len(free_neighbors)==2:
                if self.get(self.neighbor_table[point][0])==player:
                    if (self._connections.get(self._grid.get(corners[1])) is self._connections[self.edge_strings[player]]):
                        return free_neighbors
                    else:
                        return [free_neighbors[0]]
                elif self.get(self.neighbor_table[point][2])==player:
                    if self._connections.get(self._grid.get(corners[0])) is self._connections[self.edge_strings[player]]:
                        return free_neighbors
                    else:
                        return [free_neighbors[0]]
                else:
                    if (self._connections.get(self._grid.get(self.neighbor_table[point][1])) is self._connections[self.edge_strings[player]]):
                        return free_neighbors
                    else:
                        return [free_neighbors[0]]
            else:
                if  (self._connections.get(self._grid.get(corners[0])) is self._connections[self.edge_strings[player]]) and \
                     (self._connections.get(self._grid.get(corners[1])) is self._connections[self.edge_strings[player]]):
                    return free_neighbors
                elif self._connections.get(self._grid.get(corners[0])) is self._connections[self.edge_strings[player]]:
                    return [free_neighbors[0],free_neighbors[1]]
                elif self._connections.get(self._grid.get(corners[1])) is self._connections[self.edge_strings[player]]:
                    return [free_neighbors[1],free_neighbors[2]]
                elif self._connections.get(self._grid.get(corners[0])) is not None and \
                    self._connections.get(self._grid.get(corners[0])) is self._connections.get(self._grid.get(corners[1])):
                    return [free_neighbors[0],free_neighbors[1]]
                else:
                    return [free_neighbors[0]]

        region_count = 1
        not_region_count = 1
        region = [0]*8
        not_region = [0]*8

        for i in range(4):
            if self.get(self.neighbor_table[point][i]) == player:
                region[2*i] = 0
                if region[2*i-1] != 0:
                    region_count += 1
                not_region[2*i] = not_region_count
            else:
                not_region[2*i] = 0
                if not_region[2*i-1] != 0:
                    not_region_count += 1
                region[2*i] = region_count
            if self.get(self.corner_table[point][i]) == player:
                region[2*i+1] = 0
                if region[2*i] != 0:
                    region_count += 1
                not_region[2*i+1] = not_region_count
            else:
                not_region[2*i+1] = 0
                if not_region[2*i] != 0:
                    not_region_count += 1
                region[2*i+1] = region_count

        if region[0] == 1 and region[7] == 1:
            return [self.neighbor_table[point][0]]
        elif not_region[0] == 1 and not_region[7] == 1:
            return []

        if region[0] == 0 and region[7] == 0:
            i = -1
            while region[i] == 0:
                not_region[i] = 1
                i -= 1
        if not_region[0] == 0 and not_region[7] == 0:
            i = -1
            while not_region[i] == 0:
                region[i] = 1
                i -= 1

        conns = []
        for i in range(4):
            conns.append(self._connections.get(self._grid.get(self.neighbor_table[point][i])))
            conns.append(self._connections.get(self._grid.get(self.corner_table[point][i])))

        num_regions = max(region)

        conn_before = [None]*num_regions
        conn_after = [None]*num_regions
        for i in range(-1,7):
            if region[i]==0 and region[i+1]!=0:
                conn_before[region[i+1]-1] = conns[i]
            if region[i]!=0 and region[i+1]==0:
                conn_after[region[i]-1] = conns[i+1]

        distinct_regions = []
        for i in range(num_regions):

            if conn_after[i] in [conn_before[index] for index in range(i+1)]:
                distinct_regions.append(i+1)

        distinct_regions_points = []
        for dist_r in distinct_regions:
            for i in range(4):
                if region[2*i] == dist_r:
                    distinct_regions_points.append(self.neighbor_table[point][i])
                    break

        return distinct_regions_points

    def find_region(self, color, start_pos, visited=None):
        if visited is None:
            visited = {}
        if start_pos in visited:
            return set(), set()
        all_points =  {start_pos}
        all_border_strings = set()
        visited[start_pos] = True
        for neighbor in self.neighbor_table[start_pos]:
            if self.get(neighbor) != color:
                points, border_strings = self.find_region(color, neighbor, visited)
                all_points |= points
                all_border_strings |= border_strings
            else:
                all_border_strings.add(self._grid.get(neighbor))
        return all_points, all_border_strings

    def find_healthy_regions(self):
        for string, regions in self._regions_by_string.items():
            healthy_regions = set()

            for region in regions:
                healthy = True
                for region_point in region.points:
                    if self.get(region_point)==None and region_point not in self._liberties[string]:
                        healthy = False
                        break
                if healthy:
                    healthy_regions.add(region)
            self._vital_regions_by_string[string.color][string] = set().union(healthy_regions)
            self._healthy_regions_by_string[string.color][string] = set().union(healthy_regions)                             


    def find_potentially_safe_strings(self):
        for string, regions in self._vital_regions_by_string[Player.black].items():
            for region in regions:
                if region not in self._safe_strings_by_region[Player.black]:
                    self._safe_strings_by_region[Player.black][region] = set()
                if region not in self._potentially_safe_strings_by_region[Player.black]:
                    self._potentially_safe_strings_by_region[Player.black][region] = set()
                self._safe_strings_by_region[Player.black][region].add(string)
                self._potentially_safe_strings_by_region[Player.black][region].add(string)

        for string, regions in self._vital_regions_by_string[Player.white].items():
            for region in regions:
                if region not in self._safe_strings_by_region[Player.white]:
                    self._safe_strings_by_region[Player.white][region] = set()
                if region not in self._potentially_safe_strings_by_region[Player.white]:
                    self._potentially_safe_strings_by_region[Player.white][region] = set()
                self._safe_strings_by_region[Player.white][region].add(string)
                self._potentially_safe_strings_by_region[Player.white][region].add(string)

    def reduce_healthy_regions(self):
        discarded_regions = set()
        for region, strings in self._safe_strings_by_region[Player.black].items():
            if len(strings)<len(self._strings_by_region[region]):
                discarded_regions.add(region)
                for safe_string in self._vital_regions_by_string[Player.black]:
                    self._vital_regions_by_string[Player.black][safe_string] -= {region}
        for region in discarded_regions:
            del(self._safe_strings_by_region[Player.black][region])

        discarded_regions = set()
        for region, strings in self._safe_strings_by_region[Player.white].items():
            if len(strings)<len(self._strings_by_region[region]):
                discarded_regions.add(region)
                for safe_string in self._vital_regions_by_string[Player.white]:
                    self._vital_regions_by_string[Player.white][safe_string] -= {region}
        for region in discarded_regions:
            del(self._safe_strings_by_region[Player.white][region])

    def reduce_potentially_safe_strings(self):
        discarded_strings = set()
        for string, regions in self._vital_regions_by_string[Player.black].items():
            if len(regions)<2:
                discarded_strings.add(string)
                for vital_region in self._safe_strings_by_region[Player.black]:
                    self._safe_strings_by_region[Player.black][vital_region] -= {string}
        for string in discarded_strings:
            del(self._vital_regions_by_string[Player.black][string])

        discarded_strings = set()
        for string, regions in self._vital_regions_by_string[Player.white].items():
            if len(regions)<2:
                discarded_strings.add(string)
                for vital_region in self._safe_strings_by_region[Player.white]:
                    self._safe_strings_by_region[Player.white][vital_region] -= {string}
        for string in discarded_strings:
            del(self._vital_regions_by_string[Player.white][string])


    def find_safe_strings_and_vital_regions(self):
        old_num_strings = 0
        old_num_regions = 0
        self.find_healthy_regions()
        self.find_potentially_safe_strings()
        new_num_strings = len(self._vital_regions_by_string[Player.black])+len(self._vital_regions_by_string[Player.white])
        new_num_regions = len(self._safe_strings_by_region[Player.black])+len(self._safe_strings_by_region[Player.white])
        while new_num_regions != old_num_regions or new_num_strings != old_num_strings:
            self.reduce_potentially_safe_strings()
            self.reduce_healthy_regions()
            old_num_strings = new_num_strings
            old_num_regions = new_num_regions
            new_num_strings = len(self._vital_regions_by_string[Player.black])+len(self._vital_regions_by_string[Player.white])
            new_num_regions = len(self._safe_strings_by_region[Player.black])+len(self._safe_strings_by_region[Player.white])
            


    def _remove_string(self, string):

        for stone_point in string.stones:
            for neighbor in self.neighbor_table[stone_point]:
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    self.string_add_liberty(neighbor_string, stone_point)
            del(self._grid[stone_point])
            self._hash ^= zobrist.HASH_CODE[stone_point, string.color]
        for conn_string in self._connected[string]:
            self.string_delete_connected(conn_string, string)
        all_visited_connections = set()
        for conn_string in self._connected[string]:
            if conn_string not in all_visited_connections:
                found_connections = self.find_connections(conn_string)
                all_visited_connections |= found_connections
                new_connection = StringConnection(conn_string.color,found_connections)
                for visited_string in found_connections:
                    self._connections[visited_string] = new_connection
        del(self._connections[string])

        old_regions_this = self._regions_by_string.get(string)
        new_region_this_stones = string.stones
        for region in old_regions_this:
            new_region_this_stones |= region.points
        new_region_this = Region(string.color, new_region_this_stones)
        for region_point in new_region_this.points:
            self.assign_new_region_to_point(string.color, new_region_this, region_point)

        new_region_this_strings = set()
        for region in old_regions_this:
            new_region_this_strings |= self._strings_by_region.get(region)
        new_region_this_strings -= {string}

        self._strings_by_region[new_region_this] = new_region_this_strings

        for region_string in new_region_this_strings:
            self._regions_by_string[region_string] -= old_regions_this
            self._regions_by_string[region_string] |= {new_region_this}

        del(self._regions_by_string[string])
        for region in old_regions_this:
            del(self._strings_by_region[region])


    def place_stone(self, player, point):
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None

        adjacent_same_color = []
        same_color_connected = []
        same_color_connections = []
        adjacent_opposite_color = []
        liberties = []

        div_reg_result = self.divide_region(player, point)

        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            neighbor_connection = self._connections.get(neighbor_string)
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
                if neighbor_connection not in same_color_connections:
                    same_color_connections.append(neighbor_connection)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)

        for corner in self.corner_table[point]:

            corner_connected = self._grid.get(corner)
            if ((corner_connected is not None) and (corner_connected.color == player)
                and (corner_connected not in same_color_connected)):
                same_color_connected.append(corner_connected)

            corner_connection = self._connections.get(self._grid.get(corner))
            if ((corner_connection is not None) and (corner_connection.color == player)
                and (corner_connection not in same_color_connections)):
                same_color_connections.append(corner_connection)

        if len(self.neighbor_table[point]) < 4:
            same_color_connected.append(self.edge_strings[player])
            if self._connections.get(self.edge_strings[player]) not in same_color_connections:
                same_color_connections.append(self._connections.get(self.edge_strings[player]))

        affected_regions = set()
        for string in adjacent_same_color:
            affected_regions |= self._regions_by_string[string]

        new_string = GoString(player, [point])
        self._liberties[new_string] = set(liberties)
        self._connected[new_string] = set(same_color_connected)

        for same_color_string in adjacent_same_color:
            new_string = self.strings_merged(new_string, same_color_string)
        self._connected[new_string] -= set(adjacent_same_color)

        for conn_string in self._connected[new_string]:
            self._connected[conn_string] -= set(adjacent_same_color)

        for connected_string in self._connected[new_string]:
            self.string_add_connected(connected_string, new_string)

        new_string_connection = StringConnection(player,[new_string])

        for same_color_connection in same_color_connections:
            new_string_connection = new_string_connection.merged_with(same_color_connection)

        new_string_connection = new_string_connection.without_strings(adjacent_same_color)

        for same_color_string in adjacent_same_color:
            del(self._connections[same_color_string])

        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string
        for new_member_string in new_string_connection.strings:
            self._connections[new_member_string] = new_string_connection

        self._hash ^= zobrist.HASH_CODE[point, player]


        old_region = self.read_region_by_point(player, point)

        old_region_strings = self._strings_by_region.get(old_region)

        for region in (affected_regions-{old_region}):
            self._strings_by_region[region] -= set(adjacent_same_color)
            self._strings_by_region[region] |= {new_string}


        if old_region_strings==None:
            old_region_strings=set()

        if len(div_reg_result)==0:
            new_string_regions = set()
            for string in adjacent_same_color:
                new_string_regions |= self._regions_by_string[string]

            new_string_regions -= {old_region}

            self.delete_point_from_region_by_point(player, point)
            for string in adjacent_same_color:
                del(self._regions_by_string[string])
            if old_region in self._strings_by_region:
                del(self._strings_by_region[old_region])

            self._regions_by_string[new_string] = new_string_regions


        elif len(div_reg_result)==1:
            new_region = Region(player,old_region.points - {point})

            new_string_regions = set()
            for string in adjacent_same_color:
                new_string_regions |= self._regions_by_string[string]

            new_string_regions -= {old_region}
            new_string_regions |= {new_region}

            new_region_strings = old_region_strings - set(adjacent_same_color)
            new_region_strings.add(new_string)

            self.delete_point_from_region_by_point(player, point)

            if old_region in self._strings_by_region:
                del(self._strings_by_region[old_region])

            for string in old_region_strings:
                self._regions_by_string[string] -= {old_region}

            for string in adjacent_same_color:
                del(self._regions_by_string[string])

            for region_point in new_region.points:
                self.assign_new_region_to_point(player, new_region, region_point)

            self._regions_by_string[new_string] = new_string_regions

            for string in new_region_strings:
                self._regions_by_string[string] |= {new_region}

            self._strings_by_region[new_region] = new_region_strings

        else:
            new_regions = [self.find_region(player,start) for start in div_reg_result]
            self.delete_point_from_region_by_point(player, point)
            if old_region in self._strings_by_region:
                del(self._strings_by_region[old_region])
            for string in old_region_strings:
                self._regions_by_string[string] -= {old_region}

            new_string_regions = set()
            for string in adjacent_same_color:
                new_string_regions |= self._regions_by_string[string]
            new_string_regions -= {old_region}
            for string in adjacent_same_color:
                del(self._regions_by_string[string])
            self._regions_by_string[new_string] = new_string_regions
            for new_region_data in new_regions:
                new_region = Region(player,new_region_data[0])
                for region_point in new_region.points:
                    self.assign_new_region_to_point(player, new_region, region_point)
                for string in new_region_data[1]:
                    self._regions_by_string[string] |= {new_region}
                self._strings_by_region[new_region] = new_region_data[1]

        for other_color_string in adjacent_opposite_color:
            self.string_delete_liberty(other_color_string, point)
            if not self.num_liberties(other_color_string):
                self._remove_string(other_color_string)

        self.find_safe_strings_and_vital_regions()


    def is_self_capture(self, player, point):
        friendly_strings = []
        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                return False
            elif neighbor_string.color == player:
                friendly_strings.append(neighbor_string)
            else:
                if self.num_liberties(neighbor_string) == 1:
                    return False
        if all(self.num_liberties(neighbor) == 1 for neighbor in friendly_strings):
            return True
        return False

    def zobrist_hash(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, Board) and \
            self.num_rows == other.num_rows and \
            self.num_cols == other.num_cols and \
            self._grid == other._grid

    def __deepcopy__(self, memodict={}):
        copied = Board(self.num_rows, self.num_cols)
        copied._grid = copy.copy(self._grid)
        copied._liberties = copy.copy(self._liberties)
        copied._connected = copy.deepcopy(self._connected)
        copied._connections = copy.deepcopy(self._connections)
        copied._region_by_point_black = copy.copy(self._region_by_point_black)
        copied._region_by_point_white = copy.copy(self._region_by_point_white)
        copied._regions_by_string = copy.deepcopy(self._regions_by_string)
        copied._strings_by_region = copy.deepcopy(self._strings_by_region)
        copied._hash = self._hash
        return copied


class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        if previous is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states |
                {(previous.next_player, previous.board.zobrist_hash())})
        self.last_move = move

    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    @property
    def situation(self):
        return (self.next_player, self.board)

    def is_over(self):
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False
        return self.board.is_self_capture(player, move.point)

    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = self.apply_move(move).board
        next_situation = (player.other, next_board.zobrist_hash())
        ko_found = False
        if next_situation in self.previous_states:
            next_situation_board = (player.other, next_board)
            past_state = self.previous_state
            while past_state is not None and not ko_found:
                if past_state.situation == next_situation_board:
                    ko_found = True
                past_state = past_state.previous_state
        return ko_found

    def is_valid_move(self, move):
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (
            self.board.get(move.point) is None and
            not self.is_move_self_capture(self.next_player, move) and
            not self.does_move_violate_ko(self.next_player, move))

    def is_sensible_move(self, move):
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        if self.board.get(move.point) is None:
            own_region_played_in = self.board.read_region_by_point(self.next_player, move.point)
            other_region_played_in = self.board.read_region_by_point(self.next_player.other, move.point)
            if (not self.is_move_self_capture(self.next_player, move) and
                self.board._safe_strings_by_region[self.next_player].get(own_region_played_in) == None and
                self.board._safe_strings_by_region[self.next_player.other].get(other_region_played_in) == None):
                return (not self.does_move_violate_ko(self.next_player, move))
        else:
            return False

    def legal_moves(self):
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        moves.append(Move.pass_turn())
        return moves

    def sensible_legal_moves(self):
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row, col))
                if self.is_sensible_move(move):
                    moves.append(move)
        moves.append(Move.pass_turn())
        return moves

    def winner(self):
        if not self.is_over():
            return None
        if self.last_move.is_resign:
            return self.next_player
        game_result = compute_game_result(self)
        return game_result.winner

