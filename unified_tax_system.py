# unified_tax_system.py
import math
from bisect import bisect_right

class UnifiedTaxSystem:
    def __init__(self, slabs=None, multi_dim_data=None, deductions=None):
        self.slabs = slabs or []
        self.dynamic_slabs = []  # Replaced SortedList with regular list
        self.multi_dim_data = multi_dim_data or []
        self.deductions = deductions or []
        self.is_static = True
        self.dimensions = 1
        self._precompute()

    def _precompute(self):
        """Smart precomputation based on data characteristics"""
        if self.slabs:
            # Choose between prefix sum and sparse table
            if len(self.slabs) < 1000:
                # Prefix Sum (1D)
                self.slabs.sort()
                self.prefix = [0]
                for lower, upper, rate in self.slabs:
                    self.prefix.append(self.prefix[-1] + (upper - lower) * rate)
                
                # Bitmask optimization
                if len(self.slabs) < 64:
                    self.bitmask = sum(1 << i for i, (_, _, r) in enumerate(self.slabs))
                else:
                    self.bitmask = 0
            else:
                # Sparse Table (1D)
                n = len(self.slabs)
                k = math.floor(math.log2(n))
                self.st = [[0]*n for _ in range(k+1)]
                for i in range(n):
                    self.st[0][i] = self.slabs[i][2]
                for j in range(1, k+1):
                    for i in range(n - (1 << j) + 1):
                        self.st[j][i] = max(self.st[j-1][i], self.st[j-1][i + (1 << (j-1))])
            
            # Interval Tree (1D)
            self.max_end = [0]*(2*len(self.slabs))
            for i, (lower, upper, _) in enumerate(self.slabs):
                self.max_end[len(self.slabs)+i] = upper
            for i in range(len(self.slabs)-1, 0, -1):
                self.max_end[i] = max(self.max_end[2*i], self.max_end[2*i+1])
        
        # KD-Tree (Multi-D)
        if self.multi_dim_data and len(self.multi_dim_data[0]) > 2:
            self.dimensions = len(self.multi_dim_data[0]) - 1
            self.kdtree = self._build_kdtree(self.multi_dim_data)
        
        # Dynamic Programming cache
        self.dp_cache = {}

    def calculate_tax(self, income, params=None):
        """Unified tax calculation with auto-algorithm selection"""
        if params is None or not self.multi_dim_data:
            return self._1d_tax(income)
        return self._nd_tax(income, params)

    def _1d_tax(self, income):
        """Optimized 1D calculation with structure switching"""
        if self.is_static:
            if hasattr(self, 'prefix'):
                # Prefix sum calculation
                idx = bisect_right([s[0] for s in self.slabs], income) - 1
                if idx < 0:
                    return 0
                lower, upper, rate = self.slabs[idx]
                return self.prefix[idx] + max(0, (min(income, upper) - lower)) * rate
            else:
                # Sparse table lookup
                j = math.floor(math.log2(len(self.slabs)))
                max_val = max(self.st[j][0], self.st[j][len(self.slabs)-(1<<j)])
                return income * max_val
        else:
            # Dynamic structure lookup using bisect
            dummy = (income, math.inf, 0)  # Search tuple
            idx = bisect_right(self.dynamic_slabs, dummy) - 1
            if idx >= 0:
                return income * self.dynamic_slabs[idx][2]
            return 0

    def _nd_tax(self, income, params):
        """Multi-dimensional tax calculation"""
        if self.dimensions == 1:
            return self._1d_tax(income)
        
        # KD-Tree search
        best = self._kdtree_search(self.kdtree, [income] + list(params), 0)
        return best[-1] * income

    def optimize_deductions(self, max_limit):
        """Hybrid optimization using DP + bitmasking"""
        if len(self.deductions) < 20:  # Bitmask for small sets
            max_val = 0
            for mask in range(1 << len(self.deductions)):
                total = cost = 0
                for i in range(len(self.deductions)):
                    if mask & (1 << i):
                        total += self.deductions[i][0]
                        cost += self.deductions[i][1]
                if cost <= max_limit:
                    max_val = max(max_val, total)
            return max_val
        else:  # DP for larger sets
            dp = [0]*(max_limit+1)
            for val, cost in self.deductions:
                for w in range(max_limit, cost-1, -1):
                    dp[w] = max(dp[w], dp[w-cost] + val)
            return dp[max_limit]

    def update_slabs(self, new_slabs):
        """Efficient updates with structure switching"""
        if len(new_slabs) - len(self.slabs) > 10:  # Threshold for rebuild
            self.slabs = new_slabs
            self.is_static = True
            self._precompute()
        else:  # Incremental update
            self.is_static = False
            # Manual sorting instead of SortedList
            self.dynamic_slabs.extend(new_slabs)
            self.dynamic_slabs.sort(key=lambda x: x[0])  # Sort by lower bound

    # KD-Tree helper methods
    def _build_kdtree(self, points, depth=0):
        if not points:
            return None
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
        if node is None:
            return [math.inf]*(self.dimensions+1)
        
        k = self.dimensions
        axis = depth % k
        
        if target[axis] < node['point'][axis]:
            next_branch = node['left']
            opposite_branch = node['right']
        else:
            next_branch = node['right']
            opposite_branch = node['left']
        
        best = min(
            self._kdtree_search(next_branch, target, depth+1),
            node['point'],
            key=lambda x: math.dist(x[:-1], target)
        )
        
        if math.dist(node['point'][:-1], target) < math.dist(best[:-1], target):
            best = node['point']
        
        # Check opposite branch if needed
        if abs(node['point'][axis] - target[axis]) < math.dist(best[:-1], target):
            best = min(
                best,
                self._kdtree_search(opposite_branch, target, depth+1),
                key=lambda x: math.dist(x[:-1], target)
            )
        
        return best

# Usage Example
if __name__ == "__main__":
    tax_system = UnifiedTaxSystem(
        slabs=[(0, 50000, 0.1), (50000, 100000, 0.2), (100000, math.inf, 0.3)],
        multi_dim_data=[[50000, 10000, 0.2], [100000, 20000, 0.3], [150000, 30000, 0.35]],
        deductions=[(1000, 500), (2000, 1000), (3000, 1500)]
    )

    print("1D Tax:", tax_system.calculate_tax(75000))
    print("Multi-dim Tax:", tax_system.calculate_tax(75000, [12000]))
    print("Optimized Deductions:", tax_system.optimize_deductions(2000))

    # Test dynamic slab update
    tax_system.update_slabs([(0, 60000, 0.15), (60000, math.inf, 0.25)])
    print("Updated 1D Tax:", tax_system.calculate_tax(75000))