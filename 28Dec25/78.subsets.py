# @before-stub-for-debug-begin
from python3problem78 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode id=78 lang=python3
#
# [78] Subsets
#
# https://leetcode.com/problems/subsets/description/
#
# algorithms
# Medium (81.77%)
# Likes:    18885
# Dislikes: 333
# Total Accepted:    2.8M
# Total Submissions: 3.5M
# Testcase Example:  '[1,2,3]'
#
# Given an integer array nums of unique elements, return all possible subsets
# (the power set).
# 
# The solution set must not contain duplicate subsets. Return the solution in
# any order.
# 
# 
# Example 1:
# 
# 
# Input: nums = [1,2,3]
# Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
# 
# 
# Example 2:
# 
# 
# Input: nums = [0]
# Output: [[],[0]]
# 
# 
# 
# Constraints:
# 
# 
# 1 <= nums.length <= 10
# -10 <= nums[i] <= 10
# All the numbers of nums are unique.
# 
# 
#

# @lc code=start
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans = [[]]
        temp_list = []
        def rec(temp, i):
            # Base conditon
            if i >= len(nums):
                return
            # print(temp)
            temp.append(nums[i]) # [1]
            ans.append(temp.copy()) # here temp => []
            rec(temp, i + 1) 
            temp.pop()
            rec(temp, i + 1)

        rec(temp_list, 0)
        return ans
# @lc code=end

