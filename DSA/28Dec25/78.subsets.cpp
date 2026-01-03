#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;
        ans.push_back({}); // Start with empty subset
        
        for (int num : nums) {
            int n = ans.size();
            for (int i = 0; i < n; i++) {
                vector<int> newSubset = ans[i]; // Copy existing
                newSubset.push_back(num);       // Add current number
                ans.push_back(newSubset);       // Add to answer
            }
        }
        
        return ans;
    }
};
