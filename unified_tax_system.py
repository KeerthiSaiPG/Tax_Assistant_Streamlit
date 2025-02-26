# unified.py
import heapq
import math
from bisect import bisect_right
from sortedcontainers import SortedList

class UnifiedTaxSystem:
    def __init__(self, slabs=None, multi_dim_data=None, deductions=None):
        self.slabs = slabs or []
        self.dynamic_slabs = SortedList()
        self.multi_dim_data = multi_dim_data or []
        self.deductions = deductions or []
        self.is_static = True
        self.dimensions = 1
        self._precompute()

    def _precompute(self):
        if self.slabs:
            if len(self.slabs) < 1000:
                self.slabs.sort()
                self.prefix = [0]
                for lower, upper, rate in self.slabs:
                    self.prefix.append(self.prefix[-1] + (upper - lower) * rate)
                self.bitmask = sum(1 << i for i, (_, _, r) in enumerate(self.slabs) if len(self.slabs) < 64)
            else:
                n = len(self.slabs)
                k = math.floor(math.log2(n))
                self.st = [[0]*n for _ in range(k+1)]
                for i in range(n):
                    self.st[0][i] = self.slabs[i][2]
                for j in range(1, k+1):
                    for i in range(n - (1 << j) + 1):
                        self.st[j][i] = max(self.st[j-1][i], self.st[j-1][i + (1 << (j-1))])
            
            self.max_end = [0]*(2*len(self.slabs))
            for i, (lower, upper, _) in enumerate(self.slabs):
                self.max_end[len(self.slabs)+i] = upper
            for i in range(len(self.slabs)-1, 0, -1):
                self.max_end[i] = max(self.max_end[2*i], self.max_end[2*i+1])
        
        if self.multi_dim_data and len(self.multi_dim_data[0]) > 2:
            self.dimensions = len(self.multi_dim_data[0]) - 1
            self.kdtree = self._build_kdtree(self.multi_dim_data)
        
        self.dp_cache = {}

    def calculate_tax(self, income, params=None):
        if params is None or not self.multi_dim_data:
            return self._1d_tax(income)
        return self._nd_tax(income, params)

    def _1d_tax(self, income):
        if self.is_static:
            if hasattr(self, 'prefix'):
                idx = bisect_right([s[0] for s in self.slabs], income) - 1
                if idx < 0: return 0
                lower, upper, rate = self.slabs[idx]
                return self.prefix[idx] + max(0, (min(income, upper) - lower)) * rate
            else:
                j = math.floor(math.log2(len(self.slabs)))
                return income * max(self.st[j][0], self.st[j][len(self.slabs)-(1<<j)])
        else:
            idx = self.dynamic_slabs.bisect_right((income, math.inf)) - 1
            return income * self.dynamic_slabs[idx][2] if idx >= 0 else 0

    def _nd_tax(self, income, params):
        if self.dimensions == 1: return self._1d_tax(income)
        best = self._kdtree_search(self.kdtree, [income] + list(params), 0)
        return best[-1] * income

    def optimize_deductions(self, max_limit):
        if len(self.deductions) < 20:
            max_val = 0
            for mask in range(1 << len(self.deductions)):
                total = cost = 0
                for i in range(len(self.deductions)):
                    if mask & (1 << i):
                        total += self.deductions[i][0]
                        cost += self.deductions[i][1]
                if cost <= max_limit: max_val = max(max_val, total)
            return max_val
        else:
            dp = [0]*(max_limit+1)
            for val, cost in self.deductions:
                for w in range(max_limit, cost-1, -1):
                    dp[w] = max(dp[w], dp[w-cost] + val)
            return dp[max_limit]

    def update_slabs(self, new_slabs):
        if len(new_slabs) - len(self.slabs) > 10:
            self.slabs = new_slabs
            self.is_static = True
            self._precompute()
        else:
            self.is_static = False
            self.dynamic_slabs.update(new_slabs)

    def _build_kdtree(self, points, depth=0):
        if not points: return None
        k = len(points[0])-1
        axis = depth % k
        points.sort(key=lambda x: x[axis])
        mid = len(points) // 2
        return {
            'point': points[mid],
            'left': self._build_kdtree(points[:mid], depth+1),
            'right': self._build_kdtree(points[mid+1:], depth+1)
        }

    def _kdtree_search(self, node, target, depth):
        if node is None: return [math.inf]*(self.dimensions+1)
        k = self.dimensions
        axis = depth % k
        
        next_branch = node['left'] if target[axis] < node['point'][axis] else node['right']
        best = min(self._kdtree_search(next_branch, target, depth+1), node['point'],
                   key=lambda x: math.dist(x[:-1], target))
        
        if math.dist(node['point'][:-1], target) < math.dist(best[:-1], target):
            best = node['point']
        return best