#
# @lc app=leetcode id=22 lang=python3
#
# [22] Generate Parentheses
#
# https://leetcode.com/problems/generate-parentheses/description/
#
# algorithms
# Medium (77.99%)
# Likes:    22934
# Dislikes: 1065
# Total Accepted:    2.7M
# Total Submissions: 3.4M
# Testcase Example:  '3'
#
# Given n pairs of parentheses, write a function to generate all combinations
# of well-formed parentheses.
# 
# 
# Example 1:
# Input: n = 3
# Output: ["((()))","(()())","(())()","()(())","()()()"]
# Example 2:
# Input: n = 1
# Output: ["()"]
# 
# 
# Constraints:
# 
# 
# 1 <= n <= 8
# 
# 
#

# @lc code=start
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        # recursive operation. rec("" + "(", open, close)
        # start with the first bracket "(" and then rec("" + "(", open, close) and then rec("" + ")", open, close)
        # only when the open and close conditions are met.
        ans = []
        def rec (string, open, close):
            # BASE Condition
            if len(string) == 2 * n:
                # no need to return anything as we were appending the value in the ans which was global. 
                ans.append(string)
                return 
            
            # now the addition of both the brackets recursively.
            if open < n:
                rec(string + "(", open + 1, close)

            if open > close: # NOTE: IMP - here the condition is different
                rec(string + ")", open, close + 1)
        
        rec("", 0, 0)
        return ans
# @lc code=end

