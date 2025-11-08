import math
import numpy as np

"""
Histogram Data Structure from Ben-Haim and Tom-Tov (2010)

Reference: Yael Ben-Haim and Elad Tom-Tov. A streaming parallel decision tree algorithm. Journal of Ma-chine Learning Research, 11(2), 2010.
"""

class Histogram:
    def __init__(self, B: int):
        self.B = B # Max number of sets/bins
        self.bins = [] # {(p_1,m_1), ..., (p_i,m_i)}            
            
    # Algorithm 1: Update
    def update(self, p: float):
        # if p exactly matches any current p_i in the bin increase m_i
        for i, (pi, mi) in enumerate(self.bins):
            if pi == p:
                self.bins[i] = (pi, mi + 1.0)
                return
        # Otherwise add new bin 
        self.bins.append((p, 1))
        # Sort and Reduce len(bins) to B
        self._reduce_bins()
            
    # Algorithm 2: Merge self with S2
    def merge(self, S2: "Histogram"):
        # use lowest B if not the same
        if self.B != S2.B:
            self.B = self.B if self.B < S2.B else S2.B
        # Combine bins
        self.bins += S2.bins
        # Sort and Reduce len(bins) to B
        self._reduce_bins()
        
    # Algorithm 3: estimate number of points in the interval (-inf,b]
    def sum(self, b: float) -> float:
        # Assumes bins are sorted
        # find i such that pi <= b < pi+1
        i = len(self.bins) - 1 # Default to be at the last index
        for j in range(len(self.bins) - 1):
            if self.bins[j][0] <= b:
                i = j
                break
        # Set mb and s
        pi, mi = self.bins[i]
        if i == 0: # Special case b <= pi
            s = 0.0
        elif i == len(self.bins) - 1: # Special case b > pB
            s = mi # set s = mB
        else:
            pi1, mi1 = self.bins[i+1]
            mb = mi + ( (mi1 - mi) / (pi1 - pi)  ) * (b - pi)
            s = ((mi + mb) / 2) * ((b - pi) / (pi1 - pi))
        # add all mi's lower than i
        for j in range(i):
            s += self.bins[j][1]
        s += self.bins[i][1] / 2
        return float(s)
    
    # Algorithm 4: estimate median boundary values
    def uniform(self) -> list[float]:
        if len(self.bins) <= 0:
            return []
        m = np.array([x[1] for x in self.bins])
        total_sum = m.sum()
        if total_sum <= 0:
            return []
        m_gap = total_sum / self.B
        u = []
        cum_right = np.cumsum(m) - m/2
        for j in range(1, self.B - 1):
            s = j*m_gap
            # find i such that sum([-inf, p]) < s < sum([-inf, p+1])
            i = int(np.searchsorted(cum_right, s, side="right") - 1)
            i = max(0, min(i, len(m) - 2))
            s_pi = cum_right[i]
            d = s - s_pi
            pi, mi = self.bins[i]
            pi1, mi1 = self.bins[i+1]
            a = pi1 - pi
            if a <= 0:
                z = (d / mi) if mi > 0 else 0
            else:
                b = 2 * mi
                c = -2 * d
                sqrt = b**2 - (4 * a * c)
                if sqrt < 0: 
                    sqrt = 0.0
                z = (-b + math.sqrt(sqrt)) / (2 * a)
            u.append(pi + (pi1-pi) *  z)
        return u
            
        
    def _reduce_bins(self):
        # Sort bins by p
        self.bins.sort(key=lambda x:x[0])
        while len(self.bins) > self.B:
            # find i that minimized pi+1 - pi aka closest adjacent pair
            qi = 0
            qi_diff = float("inf")
            for i in range(len(self.bins) - 1):
                gap = self.bins[i+1][0] - self.bins[i][0]
                if gap < qi_diff:
                    qi = i
                    qi_diff = gap
            # Merge qi and qi+1
            bins_i = self.bins[qi]
            bins_i1 = self.bins[qi+1]
            m_new = bins_i[1] + bins_i1[1]
            p_new = (bins_i[0]*bins_i[1] + bins_i1[0]*bins_i1[1])/(m_new)
            self.bins[qi : qi + 2] = [(p_new, m_new)]
            # Sort bins by p
            self.bins.sort(key=lambda x:x[0])