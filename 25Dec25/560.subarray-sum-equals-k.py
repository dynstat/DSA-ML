#
# @lc app=leetcode id=560 lang=python3
#
# [560] Subarray Sum Equals K
#
# https://leetcode.com/problems/subarray-sum-equals-k/description/
#
# algorithms
# Medium (46.42%)
# Likes:    24402
# Dislikes: 818
# Total Accepted:    2.2M
# Total Submissions: 4.6M
# Testcase Example:  '[1,1,1]\n2'
#
# Given an array of integers nums and an integer k, return the total number of
# subarrays whose sum equals to k.
# 
# A subarray is a contiguous non-empty sequence of elements within an array.
# 
# 
# Example 1:
# Input: nums = [1,1,1], k = 2
# Output: 2
# Example 2:
# Input: nums = [1,2,3], k = 3
# Output: 2
# 
# 
# Constraints:
# 
# 
# 1 <= nums.length <= 2 * 10^4
# -1000 <= nums[i] <= 1000
# -10^7 <= k <= 10^7
# 
# 
#

# @lc code=start
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum_count = defaultdict(int)
        prefix_sum_count[0] = 1

        count = 0
        prefix_sum = 0
        for i, n in enumerate(nums):
            prefix_sum += n
            lookup = prefix_sum - k
            count += prefix_sum_count[lookup]
            prefix_sum_count[prefix_sum] += 1
        return count
# @lc code=end

