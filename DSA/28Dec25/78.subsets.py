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
        # Initialize the result list with an empty subset
        ans = [[]]
        # Temporary list to track the current subset during recursion
        temp_list = []

        def rec(temp, i):
            # Base case: if we've considered all elements, stop recursion
            if i >= len(nums):
                return
            
            # Include the current element in the subset
            temp.append(nums[i])
            # Add the current subset configuration to the final answer
            ans.append(temp.copy())
            
            # Recurse to include elements following the current one
            rec(temp, i + 1) 
            
            # Backtrack: remove the current element to explore subsets without it
            temp.pop()
            # Recurse skipping the current element to find subsets starting from the next index
            rec(temp, i + 1)

        # Start recursion from the first index
        rec(temp_list, 0)
        return ans
# @lc code=end

