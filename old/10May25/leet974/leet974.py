# Leetcode 974: Subarray Sums Divisible by K
# Given an integer array nums and an integer k, return the number of non-empty subarrays 
# that have a sum divisible by k.
# A subarray is a contiguous part of an array.
#
# Example 1:
# Input: nums = [4,5,0,-2,-3,1], k = 5
# Output: 7
# Explanation: There are 7 subarrays with a sum divisible by k = 5:
# [4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
#
# Example 2:
# Input: nums = [5], k = 9
# Output: 0
#
# Constraints:
# 1 <= nums.length <= 3 * 10^4
# -10^4 <= nums[i] <= 10^4
# 2 <= k <= 10^4

from collections import defaultdict
from typing import List

class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        # loop through the nums array with their prefixes
        # create a hashmap or dictionary with the remainder of the prefixes
        # if the remainder is already in the hashmap, add the value to the count
        # return the count
        # Time complexity: O(n)
        # Space complexity: O(n)
        
        num_count = defaultdict(int)
        num_count[0] = 1
        prefix_sum = 0
        result = 0
        for i, num in enumerate(nums):
            prefix_sum += num
            rem = prefix_sum % k
            if rem < 0:
                rem += k            
            result += num_count[rem]
            num_count[rem] = num_count.get(rem, 0) + 1
                 
        return result
    
if __name__ == "__main__":
    nums = [4,5,0,-2,-3,1]
    k = 5
    print(Solution().subarraysDivByK(nums, k))