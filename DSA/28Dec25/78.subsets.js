/**
 * @param {number[]} nums
 * @return {number[][]}
 */
var subsets = function (nums) {
    const ans = [[]];

    for (const num of nums) {
        const len = ans.length;
        for (let i = 0; i < len; i++) {
            // Clone the existing subset and add the new number
            ans.push([...ans[i], num]);
        }
    }

    return ans;
};
