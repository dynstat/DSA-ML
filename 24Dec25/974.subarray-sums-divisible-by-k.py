#
# @lc app=leetcode id=974 lang=python3
#
# [974] Subarray Sums Divisible by K
#
# https://leetcode.com/problems/subarray-sums-divisible-by-k/description/
#
# algorithms
# Medium (55.86%)
# Likes:    7740
# Dislikes: 344
# Total Accepted:    449.6K
# Total Submissions: 804.9K
# Testcase Example:  '[4,5,0,-2,-3,1]\n5'
#
# Given an integer array nums and an integer k, return the number of non-empty
# subarrays that have a sum divisible by k.
# 
# A subarray is a contiguous part of an array.
# 
# 
# Example 1:
# 
# 
# Input: nums = [4,5,0,-2,-3,1], k = 5
# Output: 7
# Explanation: There are 7 subarrays with a sum divisible by k = 5:
# [4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2,
# -3]
# 
# 
# Example 2:
# 
# 
# Input: nums = [5], k = 9
# Output: 0
# 
# 
# 
# Constraints:
# 
# 
# 1 <= nums.length <= 3 * 10^4
# -10^4 <= nums[i] <= 10^4
# 2 <= k <= 10^4
# 
# 
#

# @lc code=start
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        
        rem_count_map = defaultdict(int)
        rem_count_map[0] = 1
        # [4,5,0,-2,-3,1] , k = 5
        prefix_sum = 0
        count = 0

        for n in nums:
            prefix_sum += n

            # find the remainder (and normalize it for the negative values)
            rem = ((prefix_sum % k) + k) % k;

            count += rem_count_map[rem]
            # Since it is a defaultdict, there will be no errors of not
            # finding the key in the dict
            rem_count_map[rem] += 1

        return count

# @lc code=end

