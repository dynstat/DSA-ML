/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>

int** subsets(int* nums, int numsSize, int* returnSize, int** returnColumnSizes) {
    int totalSubsets = 1 << numsSize; // 2^n
    *returnSize = totalSubsets;
    
    // Allocate array of pointers for the result
    int** ans = (int**)malloc(totalSubsets * sizeof(int*));
    // Allocate array for column sizes
    *returnColumnSizes = (int*)malloc(totalSubsets * sizeof(int));
    
    // Iterative approach
    // We can also use bit manipulation: 0 to 2^n - 1
    // If we want to strictly follow the "Iterative/Cascading" logic:
    
    ans[0] = NULL; // Empty set
    (*returnColumnSizes)[0] = 0;
    
    int currentSize = 1;
    
    for (int i = 0; i < numsSize; i++) {
        // For each existing subset, create a new one with nums[i] added
        int n = nums[i];
        int startSize = currentSize;
        for (int j = 0; j < startSize; j++) {
            int oldLen = (*returnColumnSizes)[j];
            int newLen = oldLen + 1;
            
            // Allocate new row
            ans[currentSize] = (int*)malloc(newLen * sizeof(int));
            
            // Copy old values
            if (oldLen > 0) {
                memcpy(ans[currentSize], ans[j], oldLen * sizeof(int));
            }
            // Add new value
            ans[currentSize][oldLen] = n;
            
            // Set size
            (*returnColumnSizes)[currentSize] = newLen;
            
            currentSize++;
        }
    }
    
    return ans;
}
