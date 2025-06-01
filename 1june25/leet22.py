"""
22. Generate Parentheses
Solved
Medium
Topics
premium lock icon
Companies
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

 

Example 1:

Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
Example 2:

Input: n = 1
Output: ["()"]
 

Constraints:

1 <= n <= 8



                             "" (0, 0)
                            /         \
                        "(" (1, 0)     ✘ invalid
                       /       \
                   "((" (2,0)    "()" (1,1)
                  /     \          \
              "((("     "(()"      "()(" (2,1)
             (3,0)     (2,1)         \
              |          \           "()()" (2,2)
           "((())"      "(()("         \
           (3,1)        (3,1)          "()()(" (3,2)
             |             \              \
         "((()))"       "(()()"         "()()()" ✅
         (3,3) ✅        (3,2)              |
                          \            "()(())" ✅
                        "(()())" ✅

And so on...


"""

from typing import List

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result: List[str] = []

        def backtrack(current: str, open_count: int, close_count: int) -> None:
            # Base case: valid combination formed
            if len(current) == 2 * n:
                result.append(current)
                return

            # Add '(' if we can
            if open_count < n:
                backtrack(current + '(', open_count + 1, close_count)

            # Add ')' only if it won't break the balance
            if close_count < open_count:
                backtrack(current + ')', open_count, close_count + 1)

        backtrack("", 0, 0)
        return result



if __name__ == "__main__":
    # Create an instance of the Solution class
    solution = Solution()

    # Example input
    n = 3

    # Generate valid parentheses combinations
    output = solution.generateParenthesis(n)

    # Print the result
    print(f"Valid parentheses combinations for n = {n}:")
    for item in output:
        print(item)
