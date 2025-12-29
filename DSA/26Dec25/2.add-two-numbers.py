# @before-stub-for-debug-begin
from python3problem2 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode id=2 lang=python3
#
# [2] Add Two Numbers
#
# https://leetcode.com/problems/add-two-numbers/description/
#
# algorithms
# Medium (47.50%)
# Likes:    35684
# Dislikes: 7092
# Total Accepted:    6.5M
# Total Submissions: 13.8M
# Testcase Example:  '[2,4,3]\n[5,6,4]'
#
# You are given two non-empty linked lists representing two non-negative
# integers. The digits are stored in reverse order, and each of their nodes
# contains a single digit. Add the two numbers and return the sum as a linked
# list.
# 
# You may assume the two numbers do not contain any leading zero, except the
# number 0 itself.
# 
# 
# Example 1:
# 
# 
# Input: l1 = [2,4,3], l2 = [5,6,4]
# Output: [7,0,8]
# Explanation: 342 + 465 = 807.
# 
# 
# Example 2:
# 
# 
# Input: l1 = [0], l2 = [0]
# Output: [0]
# 
# 
# Example 3:
# 
# 
# Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
# Output: [8,9,9,9,0,0,0,1]
# 
# 
# 
# Constraints:
# 
# 
# The number of nodes in each linked list is in the range [1, 100].
# 0 <= Node.val <= 9
# It is guaranteed that the list represents a number that does not have leading
# zeros.
# 
# 
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
        # traverse l1 and create the number
        temp = l1
        num1 = 0
        place = 0
        while temp is not None:
            val = temp.val
            num1 += (10**place) * val
            place += 1
            temp = temp.next


        # traverse l2 and create the number
        temp = l2
        num2 = 0
        place = 0
        while temp is not None:
            val = temp.val
            num2 += (10**place) * val
            place += 1
            temp = temp.next
        
        ans_num = num1 + num2 

        if ans_num == 0:
            return ListNode(0)

            
        ans = ListNode(0)
        temp = ans
        # now convert the ans_num into an array(list)
        while ans_num > 0: # check later
            rem = ans_num % 10
            node = ListNode(rem)
            temp.next = node

            temp = temp.next
            ans_num //= 10
        return ans.next
# @lc code=end

