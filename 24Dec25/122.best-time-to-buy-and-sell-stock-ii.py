# @before-stub-for-debug-begin
from python3problem122 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode id=122 lang=python3
#
# [122] Best Time to Buy and Sell Stock II
#
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/
#
# algorithms
# Medium (70.44%)
# Likes:    15089
# Dislikes: 2802
# Total Accepted:    2.8M
# Total Submissions: 4M
# Testcase Example:  '[7,1,5,3,6,4]'
#
# You are given an integer array prices where prices[i] is the price of a given
# stock on the i^th day.
# 
# On each day, you may decide to buy and/or sell the stock. You can only hold
# at most one share of the stock at any time. However, you can sell and buy the
# stock multiple times on the same day, ensuring you never hold more than one
# share of the stock.
# 
# Find and return the maximum profit you can achieve.
# 
# 
# Example 1:
# 
# 
# Input: prices = [7,1,5,3,6,4]
# Output: 7
# Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit =
# 5-1 = 4.
# Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 =
# 3.
# Total profit is 4 + 3 = 7.
# 
# 
# Example 2:
# 
# 
# Input: prices = [1,2,3,4,5]
# Output: 4
# Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit =
# 5-1 = 4.
# Total profit is 4.
# 
# 
# Example 3:
# 
# 
# Input: prices = [7,6,4,3,1]
# Output: 0
# Explanation: There is no way to make a positive profit, so we never buy the
# stock to achieve the maximum profit of 0.
# 
# 
# 
# Constraints:
# 
# 
# 1 <= prices.length <= 3 * 10^4
# 0 <= prices[i] <= 10^4
# 
# 
#

# @lc code=start
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <=1:
            return 0
        buy =  prices[0]
        sell =  prices[1]

        total_profit = 0
        max_profit = 0
        last_sell = buy
        for i in range(1,len(prices)):
            sell = prices[i]
            if sell < last_sell:
                buy = sell
                last_sell = sell
                total_profit += max_profit
                max_profit = 0
                continue
            profit = sell - buy
            last_sell = sell
            max_profit = max(max_profit, profit)

        return total_profit + max_profit

# @lc code=end

