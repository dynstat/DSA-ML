# @before-stub-for-debug-begin
from python3problem1 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode id=1 lang=python3
#
# [1] Two Sum
#
# https://leetcode.com/problems/two-sum/description/
#
# algorithms
# Easy (56.71%)
# Likes:    66200
# Dislikes: 2466
# Total Accepted:    19.9M
# Total Submissions: 35.2M
# Testcase Example:  '[2,7,11,15]\n9'
#
# Given an array of integers nums and an integer target, return indices of the
# two numbers such that they add up to target.
# 
# You may assume that each input would have exactly one solution, and you may
# not use the same element twice.
# 
# You can return the answer in any order.
# 
# 
# Example 1:
# 
# 
# Input: nums = [2,7,11,15], target = 9
# Output: [0,1]
# Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
# 
# 
# Example 2:
# 
# 
# Input: nums = [3,2,4], target = 6
# Output: [1,2]
# 
# 
# Example 3:
# 
# 
# Input: nums = [3,3], target = 6
# Output: [0,1]
# 
# 
# 
# Constraints:
# 
# 
# 2 <= nums.length <= 10^4
# -10^9 <= nums[i] <= 10^9
# -10^9 <= target <= 10^9
# Only one valid answer exists.
# 
# 
# 
# Follow-up: Can you come up with an algorithm that is less than O(n^2) time
# complexity?
#

# @lc code=start
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # Input: nums = [2,7,11,15], target = 9
        # Output: [0,1]
        map_kc_vi = {}
        # first loop to create a hashmap for compliments and indexes.
        for i,n in enumerate(nums):
            map_kc_vi[n] = i

        # second loop to search the other value of the pair in map
        for i,n in enumerate(nums):
            comp = target - n
            i2 = 0
            if comp in map_kc_vi:
                if map_kc_vi[comp] == i:
                    continue
                i2 = map_kc_vi[comp]
                break
        return [i, i2]
        



# @lc code=end

