# @before-stub-for-debug-begin
from python3problem73 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode id=73 lang=python3
#
# [73] Set Matrix Zeroes
#
# https://leetcode.com/problems/set-matrix-zeroes/description/
#
# algorithms
# Medium (62.03%)
# Likes:    16734
# Dislikes: 844
# Total Accepted:    2.4M
# Total Submissions: 3.9M
# Testcase Example:  '[[1,1,1],[1,0,1],[1,1,1]]'
#
# Given an m x n integer matrix matrix, if an element is 0, set its entire row
# and column to 0's.
# 
# You must do it in place.
# 
# 
# Example 1:
# 
# 
# Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
# Output: [[1,0,1],[0,0,0],[1,0,1]]
# 
# 
# Example 2:
# 
# 
# Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
# Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
# 
# 
# 
# Constraints:
# 
# 
# m == matrix.length
# n == matrix[0].length
# 1 <= m, n <= 200
# -2^31 <= matrix[i][j] <= 2^31 - 1
# 
# 
# 
# Follow up:
# 
# 
# A straightforward solution using O(mn) space is probably a bad idea.
# A simple improvement uses O(m + n) space, but still not the best
# solution.
# Could you devise a constant space solution?
# 
# 
#

# @lc code=start
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # we will try to store the information of zeroes in the first row and first column
        # additionally, we will have to keep flags for knowing whether the row and the column would become zero or not.

        rows = len(matrix)
        cols = len(matrix[0])
        is_row_zero = False
        is_col_zero = False


        for r in matrix[0]:
            if r == 0:
                is_row_zero = True
                break

        for r in range(len(matrix)):
            if matrix[r][0] == 0:
                is_col_zero = True
          
                break
        # [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
        for r in range(1, rows):
            for c in range(1, cols):
                # print(f"[{r}][{c}] => {matrix[r][c]}")
                if matrix[r][c] == 0:
                    matrix[0][c] = 0
                    matrix[r][0] = 0
        # ffirst row loop, r = 0
        for c in range(1,  cols):
            if matrix[0][c] == 0:
                for r in range(rows):
                    matrix[r][c] = 0

        for r in range(1, rows):
            if matrix[r][0] == 0:
                for c in range(cols):
                    matrix[r][c] = 0

        if is_col_zero:
            for r in range(rows):
                matrix[r][0] = 0

        if is_row_zero:
            for c in range(cols):
                matrix[0][c] = 0

                


        

# @lc code=end
        