#
# @lc app=leetcode id=54 lang=python3
#
# [54] Spiral Matrix
#
# https://leetcode.com/problems/spiral-matrix/description/
#
# algorithms
# Medium (55.62%)
# Likes:    16970
# Dislikes: 1524
# Total Accepted:    2.2M
# Total Submissions: 4M
# Testcase Example:  '[[1,2,3],[4,5,6],[7,8,9]]'
#
# Given an m x n matrix, return all elements of the matrix in spiral order.
# 
# 
# Example 1:
# 
# 
# Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
# Output: [1,2,3,6,9,8,7,4,5]
# 
# 
# Example 2:
# 
# 
# Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# Output: [1,2,3,4,8,12,11,10,9,5,6,7]
# 
# 
# 
# Constraints:
# 
# 
# m == matrix.length
# n == matrix[i].length
# 1 <= m, n <= 10
# -100 <= matrix[i][j] <= 100
# 
# 
#

# @lc code=start
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        top, left, right, bottom = 0, 0, len(matrix[0]), len(matrix)
        ans = []
        while top != bottom and left != right: 
            # right direction
            for c in range(left, right):
                ans.append(matrix[top][c])
            top += 1

            # downward direction
            for r in range(top, bottom):
                ans.append(matrix[r][right - 1])
            right -= 1
            
            if not (top < bottom and left < right):
                break

            # left direction
            for c in range(right - 1, left - 1, -1):
                ans.append(matrix[bottom - 1][c])
            bottom -= 1

            # upward direction
            for r in range(bottom - 1, top - 1, -1):
                ans.append(matrix[r][left])
            left += 1
        return ans

# @lc code=end

