#
# @lc app=leetcode id=79 lang=python3
#
# [79] Word Search
#
# https://leetcode.com/problems/word-search/description/
#
# algorithms
# Medium (46.45%)
# Likes:    17376
# Dislikes: 744
# Total Accepted:    2.3M
# Total Submissions: 5M
# Testcase Example:  '[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]\n"ABCCED"'
#
# Given an m x n grid of characters board and a string word, return true if
# word exists in the grid.
# 
# The word can be constructed from letters of sequentially adjacent cells,
# where adjacent cells are horizontally or vertically neighboring. The same
# letter cell may not be used more than once.
# 
# 
# Example 1:
# 
# 
# Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word
# = "ABCCED"
# Output: true
# 
# 
# Example 2:
# 
# 
# Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word
# = "SEE"
# Output: true
# 
# 
# Example 3:
# 
# 
# Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word
# = "ABCB"
# Output: false
# 
# 
# 
# Constraints:
# 
# 
# m == board.length
# n = board[i].length
# 1 <= m, n <= 6
# 1 <= word.length <= 15
# board and word consists of only lowercase and uppercase English letters.
# 
# 
# 
# Follow up: Could you use search pruning to make your solution faster with a
# larger board?
# 
#

# @lc code=start
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # LOOP through all the elements of the matrix '
        # apply dfs (that checks all the conditions for the current element)
        # and applies the dfs on all the adjacent elements 
        rows = len(board)
        cols = len(board[0])

        def dfs(r,c,i):
            # if we have reached the end of the word to check
            # this would mean that we have covered/find all the elements of the word.
            # FINAL SUCCESS BASE CONDITION
            if i == len(word):
                return True
            
            # now check for all the possibilities that can cause the failure.
            if (r<0) or (c<0) or r>(rows - 1) or c > (cols - 1) or board[r][c] != word[i]:
                return False
            
            # now if we the execution has come to this point we know that the 
            # current element is matching with the required one in the "word".

            # now to prevent matching of next character element with this current one,
            # we will modify it to something else TEMPORARILY.
            org_val = board[r][c]
            board[r][c] = "#"
            status = dfs(r+1, c, i+1) or dfs(r, c+1, i+1) or dfs(r-1, c, i+1) or dfs(r, c-1, i+1)

            # restoring the original value
            board[r][c] = org_val 
            return status

        # this is to find the starting element of the word.
        for r in range(rows):
            for c in range(cols):
                # This if statement is important, because if we has written 
                # return dfs(r,c,0) directly, then it would return the status (whether True or False) of
                # dfs tried as (0,0) being the first/starting element. 
                if dfs(r,c,0):
                    return True
                else:
                    continue  ## just of representation
        return False
# @lc code=end

