#
# @lc app=leetcode id=77 lang=python3
#
# [77] Combinations
#
# https://leetcode.com/problems/combinations/description/
#
# algorithms
# Medium (73.88%)
# Likes:    8811
# Dislikes: 247
# Total Accepted:    1.3M
# Total Submissions: 1.7M
# Testcase Example:  '4\n2'
#
# Given two integers n and k, return all possible combinations of k numbers
# chosen from the range [1, n].
# 
# You may return the answer in any order.
# 
# 
# Example 1:
# 
# 
# Input: n = 4, k = 2
# Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
# Explanation: There are 4 choose 2 = 6 total combinations.
# Note that combinations are unordered, i.e., [1,2] and [2,1] are considered to
# be the same combination.
# 
# 
# Example 2:
# 
# 
# Input: n = 1, k = 1
# Output: [[1]]
# Explanation: There is 1 choose 1 = 1 total combination.
# 
# 
# 
# Constraints:
# 
# 
# 1 <= n <= 20
# 1 <= k <= n
# 
# 
#

# @lc code=start
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        temp = []
        ans = []
        # backtracking recursion method
        def rec(temp_list, i, k):
            # this would be true only if we have filled the required 
            # number of digits successfully, meaning that we have got the answer.
            if k == 0:
                # ans found
                ans.append(temp_list.copy())
                return
            
            # this can be true even when the ans is not acheived, so simply return.
            if i > n:
                return
            temp_list.append(i)
            rec(temp_list, i+1, k-1)

            # Note: if we don't write this, that would mean that we are not starting with the other digits.
            # An alternative would be to use a for loop at the top and used the above rec() in it, 
            # that way we would not have to pop and write this other recursion method again below
            temp_list.pop()
            rec(temp_list, i+1, k)
        rec(temp, 1, k)
        return ans
# @lc code=end

